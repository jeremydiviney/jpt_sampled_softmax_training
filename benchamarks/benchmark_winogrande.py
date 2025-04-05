import torch
import torch.nn as nn
from torch.amp import autocast
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple, Union  # Keep necessary types

# --- Helper Functions ---


def prepare_full_sentence_tokens(tokenizer: Tokenizer, full_text: str, max_seq_len: int) -> Optional[List[int]]:
    """
    Constructs the full sentence (context + choice), tokenizes it,
    and truncates from the LEFT if necessary to fit max_seq_len.
    Returns UNPADDED token IDs, or None if the choice is empty or
    the sequence becomes invalid after truncation.
    """

    tokenized_full = tokenizer.encode(full_text).ids

    if not tokenized_full:  # Handle empty results from tokenizer
        print(f"  Warning: Tokenization resulted in empty sequence for '{full_text[:50]}...'")
        return None

    # Truncation (from the left)
    if len(tokenized_full) > max_seq_len:
        truncated_len = len(tokenized_full) - max_seq_len
        token_ids = tokenized_full[truncated_len:]
        # Optional: Add a check to ensure *some* part of the choice remains?
        # This might be overly strict depending on the goal.
        # print(f"  Truncated: Original {len(tokenized_full)} -> Kept {len(token_ids)}")
    else:
        token_ids = tokenized_full

    return token_ids


def calculate_batch_full_sequence_loss(
    logits: torch.Tensor,  # Shape: [batch_size, seq_len, vocab_size]
    input_ids: torch.Tensor,  # Shape: [batch_size, seq_len]
    pad_token_id: int,  # Padding token ID
    loss_fn_eval: nn.Module,  # Expects reduction='none'
) -> torch.Tensor:  # Shape: [batch_size] - avg loss per item
    """
    Calculates the average cross-entropy loss over the ENTIRE non-padding
    part of each sequence in the batch.
    """
    batch_size, seq_len, vocab_size = logits.shape
    device = logits.device

    # Shift logits and labels for next-token prediction
    # Logits for predicting token at position i+1 are at index i
    # Labels are the actual tokens starting from position 1
    shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
    shift_labels = input_ids[..., 1:].contiguous()  # [batch, seq_len-1]

    # Calculate element-wise loss (output shape: [batch * (seq_len-1)])
    flat_logits = shift_logits.view(-1, vocab_size)
    flat_labels = shift_labels.view(-1)
    elementwise_loss = loss_fn_eval(flat_logits, flat_labels)

    # Reshape loss back to batch dimension
    loss_per_token = elementwise_loss.view(batch_size, seq_len - 1)  # [batch, seq_len-1]

    # Create mask to ignore padding tokens in the labels
    # Loss is calculated for position i based on label at i+1,
    # so we mask based on shift_labels (tokens from index 1 onwards)
    loss_mask = (shift_labels != pad_token_id).float()  # [batch, seq_len-1]

    # Apply mask and sum loss per item
    masked_loss = loss_per_token * loss_mask
    summed_loss = masked_loss.sum(dim=1)  # Shape: [batch_size]

    # Count number of non-padding tokens that contributed to the loss
    num_valid_tokens = loss_mask.sum(dim=1)  # Shape: [batch_size]

    # Calculate average loss per sequence
    # Avoid division by zero: if num_valid_tokens is 0, loss is infinite (or 0 if summed_loss is also 0)
    avg_loss = torch.full_like(summed_loss, float("inf"))
    # Create a mask for valid items (where num_valid_tokens > 0)
    valid_mask = num_valid_tokens > 0
    # Only calculate average loss for valid items
    avg_loss[valid_mask] = summed_loss[valid_mask] / num_valid_tokens[valid_mask]

    return avg_loss


def evaluate_winogrande(
    model: nn.Module,
    tokenizer: Tokenizer,
    loss_fn_eval: nn.Module,  # IMPORTANT: Ensure this has reduction='none'
    model_seq_len: int,
    max_samples: Optional[int] = None,
    winogrande_config: str = "winogrande_xl",  # Common config
) -> Dict[str, float]:
    """
    Evaluates Winogrande (XL) using zero-shot likelihood comparison
    based on the average loss over the ENTIRE constructed sentence for each choice.
    """
    benchmark_name = f"Winogrande_{winogrande_config}"
    print(f"\n--- Evaluating {benchmark_name} (Full Sentence Loss) ---")
    model.eval()

    # Determine device and sequence length

    device = next(model.parameters()).device

    print(f"Using device: {device}, model_seq_len: {model_seq_len}")

    # Validate loss function
    if not (hasattr(loss_fn_eval, "reduction") and loss_fn_eval.reduction == "none"):
        raise ValueError("loss_fn_eval must have reduction='none'")

    # Get padding token ID
    pad_token_id = tokenizer.token_to_id("[PAD]")

    print(f"Using Padding token ID: {pad_token_id}")

    # Load dataset
    try:
        dataset = load_dataset("winogrande", winogrande_config, split="validation", trust_remote_code=True)
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        print(f"Loaded dataset with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load dataset {benchmark_name}: {e}")
        return {f"{benchmark_name}_accuracy": 0.0, f"{benchmark_name}_evaluated_samples": 0}

    correct_predictions = 0
    total_evaluated = 0
    skipped_items_context = 0
    skipped_items_tokenization = 0

    # Use BFloat16 if available on CUDA, otherwise Float32
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    print(f"Using dtype: {dtype}")

    for item_idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {benchmark_name}")):
        sentence = item["sentence"]
        option1_text = item["option1"]
        option2_text = item["option2"]
        answer = item["answer"]  # "1" or "2"

        # Extract context
        context_parts = sentence.split("_", 1)
        if len(context_parts) != 2:
            skipped_items_context += 1
            continue  # Skip malformed items

        full_text_1 = context_parts[0] + option1_text + context_parts[1]
        full_text_2 = context_parts[0] + option2_text + context_parts[1]

        # Prepare batch inputs - pad to the max length between the two valid options
        batch_input_ids_list = []

        batch_input_ids_list.append(prepare_full_sentence_tokens(tokenizer, full_text_1, model_seq_len))

        batch_input_ids_list.append(prepare_full_sentence_tokens(tokenizer, full_text_2, model_seq_len))

        max_len = max(len(ids) for ids in batch_input_ids_list) if batch_input_ids_list else 0

        padded_batch_input_ids = []
        for ids in batch_input_ids_list:
            padding_len = max_len - len(ids)
            # Pad on the left, standard practice for causal LMs during inference/eval
            # although for full sentence loss calculation, padding side matters less
            # as long as the mask is correct. Let's stick to left padding.
            padded_ids = ([pad_token_id] * padding_len) + ids
            padded_batch_input_ids.append(padded_ids)

        input_ids_tensor = torch.tensor(padded_batch_input_ids, dtype=torch.long, device=device)

        # Run Inference

        with torch.no_grad(), autocast(device_type=device.type, dtype=dtype):
            # Assuming model returns (logits, ...) or just logits
            logits, _ = model(input_ids_tensor, True)  # Pass True for past_key_values if needed by model

            # Calculate Losses
            batch_losses = calculate_batch_full_sequence_loss(
                logits, input_ids_tensor, pad_token_id, loss_fn_eval
            )  # Tensor of avg losses for items in the batch

        # Map losses back and find best choice
        all_choice_losses = [float("inf")] * 2  # Initialize losses for option1, option2

        for i, loss in enumerate(batch_losses.cpu().numpy()):  # Move losses to CPU for easier handling
            all_choice_losses[i] = loss if not torch.isnan(torch.tensor(loss)).item() else float("inf")

        # Find minimum loss among calculated losses
        min_loss = float("inf")
        predicted_idx = -1
        for i, loss in enumerate(all_choice_losses):
            if loss < min_loss:
                min_loss = loss
                predicted_idx = i

        # Check prediction if one was made
        if predicted_idx != -1:
            try:
                true_label_idx = int(answer) - 1
                if predicted_idx == true_label_idx:
                    correct_predictions += 1
            except (ValueError, TypeError):
                print(f"Warning: Invalid label '{answer}' encountered.")
                # Don't count as correct, but still evaluated.

            total_evaluated += 1  # Increment only if a valid comparison was possible

    accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0.0
    print(f"--- Finished {benchmark_name}: Accuracy: {accuracy:.4f} ({correct_predictions}/{total_evaluated}) ---")
    print(f"Skipped items (bad context): {skipped_items_context}")
    print(f"Skipped items (tokenization/inference error): {skipped_items_tokenization}")
    model.train()  # Set model back to training mode
    return {"accuracy": accuracy, "evaluated_samples": total_evaluated}
