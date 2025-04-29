import random
import math
import re
import os
import json
import pickle
from bisect import bisect_left
from heapq import heappush, heappop

from multiprocessing import Pool

from typing import List, Tuple, Iterable


from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from torch.utils.data.dataset import Dataset

from datasets import load_dataset

TOKEN_PATTERN = re.compile(r"(\s+|\w+|[^\w\s])")

TOKEN_CORPUS_PATTERN = re.compile(r"(\n+|\w+|[^\w\s])")


def chunked(iterator, chunk_size):
    """Yield lists of up to chunk_size items from the iterator."""
    chunk = []
    for item in iterator:
        chunk.append(item)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_selection_table(data, tokenizer, dataset_name: str):
    vocab_size = len(tokenizer.get_vocab())
    cache_filename = f"meta_cache/jpt1/{dataset_name}/selection_table_vocab_{vocab_size}.pkl"

    if os.path.exists(cache_filename):
        print(f"Loading selection table from cache: {cache_filename}")
        with open(cache_filename, "rb") as f:
            lookup_table = pickle.load(f)
        return lookup_table

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(cache_filename), exist_ok=True)

    lookup_table = []
    total = 0

    batch_size = 1000
    print("Building selection table...")
    for i in range(0, len(data), batch_size):
        batch_text = data[i : i + batch_size]["text"]
        # batch encode once for each chunk
        encoded_batch = tokenizer.encode_batch_fast(batch_text)

        for encoded_row in encoded_batch:
            # Add 1 for the <EOT> token that will be appended later
            total += len(encoded_row.ids) + 1
            lookup_table.append(total)

        if i % (10000) == 0:
            print(f"build_selection_table: Processed {i}/{len(data)} items")

    print("Saving selection table to cache...")
    with open(cache_filename, "wb") as f:
        pickle.dump(lookup_table, f)

    print(f"build_selection_table completed! Total tokens (incl. EOT): {total}")

    return lookup_table


def get_or_train_tokenizer(text_corpus: str | Iterable[str], vocab_size: int, tokenizer_path: str):
    """
    Train a BPE tokenizer on the given corpus or load it from disk if already saved.

    Args:
        corpus (iterable of str): Text corpus for training.
        vocab_size (int): Desired vocabulary size.
        tokenizer_path (str): File path for saving/loading the tokenizer.

    Returns:
        Tokenizer: A Hugging Face Tokenizers object.
    """

    # Create directory for tokenizer if it doesn't exist
    os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)

    if os.path.exists(tokenizer_path):
        # Load the tokenizer if it exists
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:

        if isinstance(text_corpus, str):
            corpus = re.split(TOKEN_CORPUS_PATTERN, text_corpus)
            corpus = [w for w in corpus if w]  # Remove empty strings
        else:
            corpus = text_corpus

        # Create a new BPE tokenizer with an unknown token
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        # Use ByteLevel pre-tokenizer so that spaces and even cross-boundary merges can be learned
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

        # Define special tokens; these will be added to the vocabulary
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<EOT>"]  # Added <EOT>
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens, min_frequency=2)

        print("Training tokenizer...")
        # Train the tokenizer on the corpus iterator
        tokenizer.train_from_iterator(corpus, trainer=trainer)
        print("Tokenizer training complete.")

        # Set a decoder to handle the byte-level encoding (this will reassemble tokens correctly)
        tokenizer.decoder = decoders.ByteLevel()
        # Optionally, add a post-processor if you need specific handling (for example, GPT-2 style)
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        # Save the trained tokenizer to file for future use
        tokenizer.save(tokenizer_path)
        print(f"Trained and saved tokenizer to {tokenizer_path}")
        # os._exit(0)

    # Verify EOT token exists after load/train
    if tokenizer.token_to_id("<EOT>") is None:
        raise ValueError("Failed to add or load '<EOT>' token in the tokenizer.")

    return tokenizer


def load_hf_dataset(dataset_name: str):
    cache_path = f"data_cache/{dataset_name}"
    # Create cache directory if it doesn't exist
    os.makedirs(cache_path, exist_ok=True)

    try:
        # Try to load from cache first
        print(f"Loading dataset '{dataset_name}' from Hugging Face...")
        # Using the 10B sample as specified in the original code
        dataset = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", cache_dir=cache_path)
        print("Dataset loaded.")
    except Exception as e:
        print(f"Error: Could not load dataset '{dataset_name}' from Hugging Face: {e}")
        # Fallback or error handling could be added here if necessary
        raise  # Re-raise the exception if loading fails

    return dataset


class Fineweb10BDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        data_stride: int,
        hf_dataset: Dataset,
        tokenizer: Tokenizer | None,
        dataset_name: str,
        dset_ratio: float = 1.0,  # control the ratio of the dataset to use
        type: str = "train",
        cache_size: int = 5000,
    ):
        # Load TinyShakespeare from Hugging Face
        self.hf_dataset = hf_dataset
        self.seq_len = seq_len
        self.data_stride = data_stride
        self.tokenizer = tokenizer
        self.train_ratio = 4000  # Ratio used to split train/val from the master dataset indices
        self.dset_ratio = dset_ratio
        self.type = type
        self.cache_size = cache_size  # Number of tokenized documents to keep in memory
        self.cache = {}  # Dictionary to store cached tokens {data_row_idx: token_ids}
        self.cache_priority = []  # Min-heap to track priorities (token_length, data_row_idx) for cache eviction
        self.token_counts = []
        self.data_stride_indices = []

        self.pad_token_id = self.tokenizer.token_to_id("[PAD]")
        self.eot_token_id = self.tokenizer.token_to_id("<EOT>")
        if self.pad_token_id is None:
            raise ValueError("'<PAD>' token not found in tokenizer vocabulary.")
        if self.eot_token_id is None:
            raise ValueError("'<EOT>' token not found in tokenizer vocabulary.")

        # Build or load the table mapping token index to document index and offset
        self.selection_table_master = build_selection_table(self.hf_dataset["train"], self.tokenizer, dataset_name)

        self.selection_table = []
        self.selection_table_indices = []

        self.stride_indices = []

        # --- Splitting and Filtering Logic ---
        master_array = np.array(self.selection_table_master)
        # Calculate token counts for each document (including the EOT token added in build_selection_table)
        token_counts = np.diff(master_array, prepend=0)
        # Create an array of indices
        indices = np.arange(len(master_array))

        # 1. Apply train/validation split based on index modulo train_ratio
        if self.type == "train":
            split_mask = (indices % self.train_ratio) != 0
        elif self.type == "validation":
            split_mask = (indices % self.train_ratio) == 0
        else:
            raise ValueError(f"Invalid dataset type: {self.type}")

        # 2. Apply dataset ratio filtering *only* for the training set
        if self.dset_ratio < 1.0 and self.type == "train":
            # Use a fixed seed for deterministic selection
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            # Create random mask with probability = dset_ratio
            ratio_mask = rng.random(len(indices)) < self.dset_ratio
            # Combine the split mask and ratio mask: an item must satisfy both
            final_mask = np.logical_and(split_mask, ratio_mask)
            print(f"Applying dset_ratio {self.dset_ratio}: retaining {np.sum(final_mask)}/{np.sum(split_mask)} training documents.")
        else:
            # For validation or if dset_ratio is 1.0, just use the split mask
            final_mask = split_mask
            if self.type == "validation":
                print(f"Retaining {np.sum(final_mask)} validation documents.")
            else:
                print(f"Using full training split ({np.sum(final_mask)} documents).")

        # Apply the final mask to get the indices of the documents we will use
        self.selection_table_indices = indices[final_mask]

        self.token_counts = token_counts[final_mask]

        self.stride_counts = 1 + (self.token_counts - 1) // self.data_stride
        self.selection_table = np.cumsum(self.stride_counts)

        self.token_list = tokenizer.get_vocab()
        self.token_count = self.selection_table[-1].item()

    def get_data_chunk(self, idx: int):

        pre_data_row_idx = bisect_left(self.selection_table, idx)

        relative_data_chunk = idx - self.selection_table[pre_data_row_idx]

        data_row_idx = int(self.selection_table_indices[pre_data_row_idx])  # lookup the master data index for the row

        # print(f"get_data_chunk: {idx}, self.selection_table: {len(self.selection_table)}, data_row_idx: {data_row_idx}")

        # Check if the current index is in the cache
        if data_row_idx in self.cache:
            full_idx_tokens = self.cache[data_row_idx]
        else:
            # Get the full text from the dataset
            full_text = self.hf_dataset["train"][data_row_idx]["text"]
            full_idx_tokens = self.tokenizer.encode_batch_fast([full_text])[0].ids

            # Append the EOT token to the document
            full_idx_tokens = full_idx_tokens + [self.eot_token_id]

            # Cache management
            token_length = len(full_idx_tokens)

            if len(self.cache) < self.cache_size:
                # Add to cache if not full
                self.cache[data_row_idx] = full_idx_tokens
                heappush(self.cache_priority, (token_length, data_row_idx))
            elif token_length > self.cache_priority[0][0]:
                # Replace smallest item if current is larger
                smallest_length, smallest_idx = heappop(self.cache_priority)
                del self.cache[smallest_idx]
                self.cache[data_row_idx] = full_idx_tokens
                heappush(self.cache_priority, (token_length, data_row_idx))

        # Calculate start and end positions for the chunk
        start_pos = relative_data_chunk * self.data_stride

        end_pos = min(start_pos + self.seq_len, len(full_idx_tokens))

        if (end_pos - start_pos) < self.seq_len:
            start_pos = max(0, end_pos - self.seq_len)

        # Extract the chunk, handling potential end of text
        idx_chunk = full_idx_tokens[start_pos:end_pos]
        return idx_chunk

    def __len__(self) -> int:

        return self.selection_table[-1]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        token_sequence_ids = np.array(self.get_data_chunk(idx))

        # Calculate padding needed
        padding_needed = (self.seq_len + 1) - len(token_sequence_ids)

        if padding_needed > 0:
            # Create padding array filled with pad_token
            padding = np.full((padding_needed), self.pad_token_id)
            # Concatenate the token sequence with padding (pad at the end to avoid attending to the pad tokens)
            token_sequence_ids = np.concatenate([token_sequence_ids, padding], axis=0)

        x = token_sequence_ids[:-1]
        y = token_sequence_ids[1:]

        return x, y
