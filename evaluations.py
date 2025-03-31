import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tqdm import tqdm
from enum import Enum
import math
from typing import Dict, Any, Optional, List, Union

# Assume your JPT1 model and inference_step function are defined elsewhere
# from models.jpt1 import JPT1, JPT1ModelType
# from training_script import inference_step # Or wherever it's defined

# --- Configuration ---


class BenchmarkName(Enum):
    HELLASWAG = "hellaswag"
    WINOGRANDE = "winogrande"
    PIQA = "piqa"
    ARC_EASY = "arc_easy"
    # STORYCLOZE = "storycloze" # Often needs specific handling/download
    COPA = "copa"
    BOOLQ = "boolq"


# Define configuration for each benchmark
# This tells the evaluator how to load and parse each dataset
BENCHMARK_CONFIGS: Dict[BenchmarkName, Dict[str, Any]] = {
    BenchmarkName.HELLASWAG: {
        "dataset_id": "hellaswag",
        "config": None,
        "split": "validation",
        "context_key": "ctx",
        "choices_key": "endings",  # List of strings
        "label_key": "label",  # String representing integer index ("0", "1", ...)
        "label_mapping": lambda x: int(x) if x else -1,  # Convert string label to int
    },
    BenchmarkName.WINOGRANDE: {
        "dataset_id": "winogrande",
        "config": "winogrande_xl",  # Using the larger version
        "split": "validation",
        # Winogrande needs special formatting: sentence with placeholder "_"
        # option1/option2 fill the placeholder.
        "context_key": "sentence",
        "choices_key": ["option1", "option2"],  # Keys for the two choices
        "label_key": "answer",  # String "1" or "2"
        "label_mapping": lambda x: int(x) - 1 if x else -1,  # Map "1"/"2" to 0/1
        "format_func": lambda ctx, choice: ctx.replace("_", choice),  # Custom formatting
    },
    BenchmarkName.PIQA: {
        "dataset_id": "piqa",
        "config": None,
        "split": "validation",
        "context_key": "goal",
        "choices_key": ["sol1", "sol2"],  # Keys for the two choices
        "label_key": "label",  # Integer 0 or 1
        "label_mapping": lambda x: int(x) if x is not None else -1,
    },
    BenchmarkName.ARC_EASY: {
        "dataset_id": "ai2_arc",
        "config": "ARC-Easy",
        "split": "validation",
        "context_key": "question",
        "choices_key": "choices",  # Dict with 'text' (list) and 'label' (list "A","B"..)
        "label_key": "answerKey",  # Correct label e.g., "A", "1", "C"
        "label_mapping": lambda x, choice_labels: choice_labels.index(x) if x in choice_labels else -1,  # Map label ("A","B"..) to index
    },
    # Example for COPA (requires slightly different text formatting)
    BenchmarkName.COPA: {
        "dataset_id": "super_glue",
        "config": "copa",
        "split": "validation",
        # Context needs question ('cause'/'effect') and premise
        "context_key": ["premise", "question"],
        "choices_key": ["choice1", "choice2"],
        "label_key": "label",  # Integer 0 or 1
        "label_mapping": lambda x: int(x) if x is not None else -1,
        # Custom formatting based on cause/effect question
        "format_func": lambda ctx_parts, choice: f"{ctx_parts[0]} {'because' if ctx_parts[1] == 'cause' else 'so'} {choice}",
    },
    BenchmarkName.BOOLQ: {
        "dataset_id": "boolq",
        "config": None,
        "split": "validation",
        "context_key": ["passage", "question"],  # Combine passage and question
        "choices_key": [" Yes", " No"],  # Fixed choices (note leading space)
        "label_key": "answer",  # Boolean True/False
        "label_mapping": lambda x: 0 if x else 1,  # Map True->0 ('Yes'), False->1 ('No')
        "format_func": lambda ctx_parts, choice: f"{ctx_parts[0]}\nQuestion: {ctx_parts[1]}\nAnswer:{choice}",
    },
    # Add StoryCloze config here if you set up the dataset
}

# --- Core Likelihood Calculation ---


def calculate_span_loss(
    model: nn.Module,
    tokenizer: Tokenizer,
    input_ids: torch.Tensor,  # Shape: [1, seq_len]
    choice_start_token_index: int,
    device: str,
    loss_fn_eval: nn.Module,
    seq_len: int,
) -> float:
    """
    Calculates the average cross-entropy loss ONLY for the choice part of input_ids.
    Assumes input_ids is already padded/truncated to seq_len.
    """
    if choice_start_token_index >= seq_len - 1:  # No space for even one choice token's prediction
        # This can happen if context fills the whole seq_len or choice starts beyond
        return float("inf")

    with torch.no_grad(), autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
        # Ensure do_final_projection=True for standard loss calculation over vocab
        # Assuming your inference_step returns (logits, pre_output)
        # We only need logits here.
        logits, _ = inference_step(model, input_ids, do_final_projection=True)
        # logits shape: [1, seq_len, vocab_size]

    # Shift logits and labels for standard next-token prediction loss
    # Predict token at position i+1 using hidden state at position i
    shift_logits = logits[..., :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
    shift_labels = input_ids[..., 1:].contiguous()  # [1, seq_len-1]

    # Flatten for loss calculation
    # [seq_len-1, vocab_size]
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    # [seq_len-1]
    flat_labels = shift_labels.view(-1)

    # Calculate loss ONLY for the tokens corresponding to the choice
    # The choice starts at `choice_start_token_index` in the original `input_ids`.
    # In the shifted labels, we care about predictions starting from index `choice_start_token_index - 1`.
    loss_mask = torch.zeros_like(flat_labels, dtype=torch.bool)
    # Inclusive start index in the *shifted* sequence. Max with 0 for safety.
    effective_choice_start_shifted = max(0, choice_start_token_index - 1)
    # Exclusive end index
    effective_choice_end_shifted = flat_labels.size(0)  # Go up to the end

    if effective_choice_start_shifted < effective_choice_end_shifted:
        loss_mask[effective_choice_start_shifted:effective_choice_end_shifted] = True

    # Count actual tokens in the choice span for averaging later
    num_choice_tokens = loss_mask.sum().item()

    if num_choice_tokens == 0:
        # This means the choice was effectively empty after shifting/masking,
        # or started beyond the sequence length used for prediction.
        return float("inf")

    # Create labels for CrossEntropyLoss, ignoring non-choice tokens
    masked_labels = torch.where(loss_mask, flat_labels, torch.tensor(loss_fn_eval.ignore_index).to(device))

    # Calculate loss across all tokens (loss_fn ignores the masked ones)
    total_loss = loss_fn_eval(flat_logits, masked_labels)

    # Average loss *per choice token*
    # Multiply total loss by total number of elements (flat_logits.size(0))
    # because CrossEntropyLoss with reduction='mean' averages over all non-ignored elements.
    # Then divide by the actual number of choice tokens we care about.
    average_loss_per_choice_token = total_loss * (flat_logits.size(0) / num_choice_tokens)

    return average_loss_per_choice_token.item()


# --- Tokenization and Formatting Helper ---
def prepare_input_and_find_choice_start(
    tokenizer: Tokenizer, context: str, choice: str, max_seq_len: int
) -> Optional[tuple[list[int], int]]:
    """
    Tokenizes context and choice, combines them, handles truncation,
    and returns token IDs and the start index of the choice tokens.

    Returns None if the choice cannot fit or inputs are invalid.
    """
    # Tokenize context and choice separately to accurately find start index
    # Add a space before choice if context isn't empty, common practice
    # prefix = " " if context else ""
    # Using a space prefix seems standard for many implementations
    choice_prefix = " "
    tokenized_context = tokenizer.encode(context).ids
    tokenized_choice = tokenizer.encode(choice_prefix + choice).ids

    # Combine
    full_token_ids = tokenized_context + tokenized_choice

    # Calculate choice start index (it's simply the length of the context tokens)
    choice_start_index = len(tokenized_context)

    # Truncation (prioritize keeping the choice)
    if len(full_token_ids) > max_seq_len:
        choice_len = len(tokenized_choice)
        keep_context_len = max_seq_len - choice_len
        if keep_context_len < 0:
            # Choice itself is too long, cannot evaluate this sample
            print(f"Warning: Choice is longer than max_seq_len ({choice_len} > {max_seq_len}). Skipping.")
            return None

        # Truncate context from the beginning
        truncated_context_tokens = tokenized_context[-keep_context_len:]
        full_token_ids = truncated_context_tokens + tokenized_choice
        # New choice start index after context truncation
        choice_start_index = len(truncated_context_tokens)  # or keep_context_len

    # Padding (if needed, though usually handled by DataLoader or later)
    # For this item-by-item approach, we might not need explicit padding here
    # if the model handles variable lengths up to max_seq_len, but let's assume
    # fixed inputs are easier. Pad to max_seq_len.
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if not pad_token_id:
        print("Warning: [PAD] token not found in tokenizer.")
        pad_token_id = 0  # Fallback, might be incorrect

    padding_needed = max_seq_len - len(full_token_ids)
    if padding_needed > 0:
        full_token_ids = full_token_ids + ([pad_token_id] * padding_needed)
    elif padding_needed < 0:
        # This should not happen if truncation logic is correct
        print(f"Error: Still too long after truncation: {len(full_token_ids)}")
        full_token_ids = full_token_ids[:max_seq_len]

    # Ensure the start index is valid after padding/truncation
    if choice_start_index >= max_seq_len:
        # This means the choice starts at or after the max length, effectively not included
        # Or context filled the whole space.
        # print(f"Warning: Choice start index {choice_start_index} >= max_seq_len {max_seq_len}. Skipping choice.")
        return None  # Indicate failure for this choice

    return full_token_ids, choice_start_index


# --- Main Evaluation Function ---


def run_evaluation(
    model: nn.Module,
    benchmark: Union[BenchmarkName, str],
    tokenizer: Tokenizer,
    device: str,
    loss_fn_eval: nn.Module,  # Standard CE loss (ignore_index=PAD_ID)
    max_samples: Optional[int] = None,
    # batch_size: int = 8 # Batching TBD, simpler item-by-item first
    model_seq_len: Optional[int] = None,  # Get from model if possible
) -> Dict[str, float]:
    """
    Runs zero-shot likelihood evaluation on a specified benchmark.

    Args:
        model: The language model to evaluate.
        benchmark: The BenchmarkName enum value or its string name.
        tokenizer: The tokenizer used by the model.
        device: The device to run evaluation on ('cuda:0', 'cpu', etc.).
        loss_fn_eval: A standard nn.CrossEntropyLoss instance.
        max_samples: Maximum number of samples to evaluate from the dataset.
        model_seq_len: The maximum sequence length the model accepts.

    Returns:
        A dictionary containing the accuracy metric for the benchmark.
    """
    if isinstance(benchmark, str):
        try:
            benchmark_enum = BenchmarkName(benchmark)
        except ValueError:
            raise ValueError(f"Unknown benchmark string: {benchmark}. Supported: {[e.name for e in BenchmarkName]}")
    elif isinstance(benchmark, BenchmarkName):
        benchmark_enum = benchmark
    else:
        raise TypeError("benchmark must be a BenchmarkName enum or string.")

    if benchmark_enum not in BENCHMARK_CONFIGS:
        raise NotImplementedError(f"Configuration for benchmark {benchmark_enum.name} is not defined.")

    config = BENCHMARK_CONFIGS[benchmark_enum]

    if model_seq_len is None:
        try:
            # Attempt to get seq_len from the model if it has the attribute
            raw_model = model.module if hasattr(model, "module") else model
            model_seq_len = raw_model.seq_len
        except AttributeError:
            raise ValueError("model_seq_len must be provided or model must have a 'seq_len' attribute.")

    print(f"\n--- Evaluating {benchmark_enum.name} ---")
    model.eval()  # Set model to evaluation mode

    # Load dataset
    try:
        dataset = load_dataset(
            config["dataset_id"], config["config"], split=config["split"], streaming=False  # Load fully for slicing if max_samples is set
        )
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

    except Exception as e:
        print(f"Failed to load dataset {config['dataset_id']} ({config['config']}): {e}")
        return {f"{benchmark_enum.name}_accuracy": 0.0, f"{benchmark_enum.name}_evaluated_samples": 0}

    correct_predictions = 0
    total_evaluated = 0

    # Custom formatting functions from config
    format_func = config.get("format_func")
    label_mapping = config.get("label_mapping")

    # Iterate item by item (simplest implementation)
    for item in tqdm(dataset, desc=f"Evaluating {benchmark_enum.name}"):
        # Extract context
        ctx_key = config["context_key"]
        if isinstance(ctx_key, list):
            context = [item[k] for k in ctx_key]  # For BoolQ, COPA etc.
        else:
            context = item[ctx_key]

        # Extract choices
        choices_key = config["choices_key"]
        if isinstance(choices_key, list) and isinstance(choices_key[0], str) and choices_key[0] in item:
            # List of keys in the item dict (e.g., PIQA, Winogrande, COPA)
            choices = [item[k] for k in choices_key]
        elif isinstance(choices_key, str) and choices_key in item and isinstance(item[choices_key], list):
            # Key points to a list within the item (e.g., HellaSwag)
            choices = item[choices_key]
        elif isinstance(choices_key, str) and choices_key in item and isinstance(item[choices_key], dict):
            # Key points to a dict with 'text' and 'label' (e.g., ARC)
            choices_data = item[choices_key]
            choices = choices_data["text"]
            choice_labels = choices_data.get("label")  # Needed for ARC label mapping
        elif isinstance(choices_key, list) and not isinstance(choices_key[0], str):
            # Fixed list of choices (e.g., BoolQ [" Yes", " No"])
            choices = choices_key
        else:
            print(f"Warning: Could not parse choices for item: {item}. Skipping.")
            continue

        # Extract and map label
        label_key = config["label_key"]
        raw_label = item.get(label_key)

        if raw_label is None:
            print(f"Warning: Label key '{label_key}' not found or is None in item: {item}. Skipping.")
            continue

        try:
            # Handle ARC's label mapping needing choice_labels
            if benchmark_enum == BenchmarkName.ARC_EASY:
                true_label_idx = label_mapping(raw_label, choice_labels)
            elif label_mapping:
                true_label_idx = label_mapping(raw_label)
            else:
                true_label_idx = int(raw_label)  # Default attempt

            if true_label_idx < 0 or true_label_idx >= len(choices):
                print(f"Warning: Mapped label index {true_label_idx} out of bounds for choices {choices}. Raw: {raw_label}. Skipping.")
                continue
        except Exception as e:
            print(f"Warning: Could not map label '{raw_label}' for item {item}. Error: {e}. Skipping.")
            continue

        choice_losses = []
        min_loss = float("inf")
        predicted_idx = -1

        # --- Process each choice for the current item ---
        for i, choice_text in enumerate(choices):
            # Apply custom formatting if needed
            if format_func:
                # Assumes format_func takes context (or list of context parts) and choice_text
                formatted_context = format_func(context, choice_text)
                # For likelihood, we need the text *before* the choice and the choice itself
                # This is tricky with format_func. A simpler approach for non-completion tasks:
                # Treat the *entire* formatted string as the sequence and calculate its *overall* avg loss?
                # Let's stick to the "completion" idea: separate context and choice text.
                # Rework formatting for completion style:
                if benchmark_enum == BenchmarkName.WINOGRANDE:
                    # Context is the sentence up to placeholder, choice fills it
                    # We need context *before* the choice text for span loss.
                    # This requires finding the placeholder's position. Simpler:
                    # Calculate loss over the *whole formatted sentence*? Less standard.
                    # Let's assume format_func provides the *full* sequence for now.
                    # We'll calculate loss over the whole thing - simpler but deviates slightly.
                    full_text = formatted_context
                    # Use a dummy context/choice separation for prepare_input
                    eval_context = ""
                    eval_choice = full_text
                elif benchmark_enum == BenchmarkName.COPA:
                    # Context is premise + connector, choice is the cause/effect
                    connector = f" {'because' if context[1] == 'cause' else 'so'}"
                    eval_context = context[0] + connector
                    eval_choice = " " + choice_text  # Add space like other choices
                elif benchmark_enum == BenchmarkName.BOOLQ:
                    # Context is passage+question+answer prompt, choice is Yes/No
                    eval_context = f"{context[0]}\nQuestion: {context[1]}\nAnswer:"
                    eval_choice = choice_text  # Already has leading space
                else:
                    # Default: context is the primary context, choice is appended
                    eval_context = context
                    eval_choice = choice_text

            else:  # Standard completion tasks (Hellaswag, PIQA, ARC)
                eval_context = context
                eval_choice = choice_text

            # Tokenize, truncate, and find choice start index
            prepared = prepare_input_and_find_choice_start(tokenizer, eval_context, eval_choice, model_seq_len)

            if prepared is None:
                # Couldn't process this choice (e.g., too long)
                choice_losses.append(float("inf"))
                continue

            input_ids_list, choice_start_idx = prepared
            input_tensor = torch.tensor([input_ids_list], device=device)  # Add batch dim [1, seq_len]

            # Calculate loss for the choice span
            loss = calculate_span_loss(model, tokenizer, input_tensor, choice_start_idx, device, loss_fn_eval, model_seq_len)
            choice_losses.append(loss)

            # print(f"  Choice {i} Loss: {loss:.4f}") # Debugging

            if loss < min_loss:
                min_loss = loss
                predicted_idx = i
        # --- End choice processing ---

        if predicted_idx != -1:
            # print(f"Item: {item}\nPred: {predicted_idx}, True: {true_label_idx}, Losses: {choice_losses}") # Debugging
            if predicted_idx == true_label_idx:
                correct_predictions += 1
        else:
            print(f"Warning: Could not get a valid prediction for item {item}. Min loss was inf.")

        total_evaluated += 1
    # --- End dataset iteration ---

    accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0
    print(f"--- Finished {benchmark_enum.name} ---")
    print(f"Accuracy: {accuracy:.4f} ({correct_predictions}/{total_evaluated})")

    model.train()  # Set model back to training mode if applicable

    return {f"{benchmark_enum.name}_accuracy": accuracy, f"{benchmark_enum.name}_evaluated_samples": total_evaluated}


# --- Example Usage ---

if __name__ == "__main__":
    # This is a dummy example. Replace with your actual model loading and setup.
    # Assume 'model', 'tokenizer', 'device' are loaded and configured.
    # Assume 'loss_fn_eval' is a standard CrossEntropyLoss instance.

    # Dummy model placeholder
    class DummyModel(nn.Module):
        def __init__(self, vocab_size=50304, embed_dim=128, seq_len=64):
            super().__init__()
            self.seq_len = seq_len
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.linear = nn.Linear(embed_dim, vocab_size)
            print(f"DummyModel initialized with seq_len={seq_len}")

        # Simplified inference_step function for the dummy model
        def forward(self, x, do_final_projection=True):
            # x shape: [batch, seq_len]
            embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
            # Dummy processing - just return embeddings as pre_output
            pre_output = embedded
            if do_final_projection:
                logits = self.linear(embedded)  # [batch, seq_len, vocab_size]
                return logits, pre_output
            else:
                # If not projecting, return None for logits maybe?
                # Or return the embeddings directly as 'logits' if that's expected
                return embedded, pre_output  # Adjust based on inference_step needs

    # Dummy inference_step (replace with your actual one)
    def inference_step(model, x, do_final_projection):
        # In a real scenario, this calls your model's forward pass
        return model(x, do_final_projection)

    # --- Setup ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load a real tokenizer
    try:
        # Make sure you have run the main script first to create tokenizer cache
        # Update dataset_name and vocab_size if different
        dataset_name = "fineweb-10BT-edu"
        vocab_size = 50304
        tokenizer_path = f"tokenizer_cache/{dataset_name}_tokenizer_{vocab_size}.json"
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]", length=128)  # Example length
    except Exception as e:
        print(f"Error loading tokenizer: {e}. Evaluations require a valid tokenizer.")
        exit()

    # Dummy model needs vocab size from tokenizer
    model_seq_len = 128  # Set based on your model/tokenizer padding
    model = DummyModel(vocab_size=tokenizer.get_vocab_size(), seq_len=model_seq_len).to(device)

    # Standard loss function for evaluation (ignore padding)
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        print("Error: [PAD] token ID not found in tokenizer!")
        pad_token_id = -100  # Common practice, but check your setup
    loss_fn_eval = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    # --- Run Evaluations ---
    results = {}
    max_eval_samples = 100  # Use a small number for quick testing

    for benchmark_name in BenchmarkName:
        # Skip benchmarks that might fail with dummy setup if needed
        # if benchmark_name in []: continue

        eval_result = run_evaluation(
            model=model,
            benchmark=benchmark_name,
            tokenizer=tokenizer,
            device=device,
            loss_fn_eval=loss_fn_eval,
            max_samples=max_eval_samples,
            model_seq_len=model_seq_len,
        )
        results.update(eval_result)

    print("\n--- Final Evaluation Results ---")
    import json

    print(json.dumps(results, indent=2))
