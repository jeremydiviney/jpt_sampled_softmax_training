import torch
import torch.nn as nn
from torch.amp import autocast
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict


def prepare_tokens_and_choice_start(
    tokenizer: Tokenizer, context: str, choice: str, max_seq_len: int
) -> Optional[Tuple[List[int], int, int]]:
    """
    Tokenizes context and choice, combines, truncates prioritizing choice.
    Returns UNPADDED token IDs, the start index of choice tokens,
    and the number of choice tokens. Returns None if invalid (e.g., choice too long).
    """
    choice_prefix = " "  # Standard prefix
    tokenized_context = tokenizer.encode(context).ids
    # Tokenize choice with prefix
    tokenized_choice = tokenizer.encode(choice_prefix + choice).ids
    choice_len = len(tokenized_choice)

    full_token_ids = tokenized_context + tokenized_choice
    choice_start_index = len(tokenized_context)

    # Truncation (prioritize keeping the choice)
    if len(full_token_ids) > max_seq_len:
        keep_context_len = max_seq_len - choice_len
        if keep_context_len < 0:
            # Choice itself is longer than max_seq_len
            # Option 1: Truncate choice (might lose meaning) - let's return None
            # Option 2: Return None (safer)
            return None  # Choice too long to fit even alone

        # Truncate context from the left
        truncated_context_tokens = tokenized_context[-keep_context_len:]
        full_token_ids = truncated_context_tokens + tokenized_choice
        choice_start_index = len(truncated_context_tokens)  # Recalculate start index

    # Ensure choice_start_index is valid within the final sequence length
    if choice_start_index >= len(full_token_ids):
        # This can happen if context gets truncated entirely and choice starts immediately
        # Or if the choice itself was truncated to empty (handled by keep_context_len < 0)
        # Let's consider it invalid if the effective choice start is at or beyond seq len
        if choice_start_index >= max_seq_len:
            return None

    # Return unpadded tokens, choice start index, and choice length
    return full_token_ids, choice_start_index, choice_len


def calculate_batch_span_loss(
    logits: torch.Tensor,  # Shape: [batch_size, seq_len, vocab_size]
    input_ids: torch.Tensor,  # Shape: [batch_size, seq_len]
    choice_start_indices: List[int],  # List of start indices, len = batch_size
    choice_lengths: List[int],  # List of choice lengths, len = batch_size
    loss_fn_eval: nn.Module,  # Expects reduction='none'
    device: str,
) -> torch.Tensor:  # Shape: [batch_size] - loss per item
    """
    Calculates the average cross-entropy loss ONLY for the choice part
    for each item in the batch.
    """
    batch_size, seq_len, _ = logits.shape

    # Shift logits and labels for next token prediction
    # Logits shape: [batch_size, seq_len-1, vocab_size]
    # Labels shape: [batch_size, seq_len-1]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()

    # Flatten logits and labels for loss calculation
    # Flat logits shape: [batch_size * (seq_len-1), vocab_size]
    # Flat labels shape: [batch_size * (seq_len-1)]
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    # Calculate element-wise loss (reduction='none')
    # elementwise_loss shape: [batch_size * (seq_len-1)]
    elementwise_loss = loss_fn_eval(flat_logits, flat_labels)

    # Reshape loss back to batch dimension
    # loss_per_token shape: [batch_size, seq_len-1]
    loss_per_token = elementwise_loss.view(batch_size, seq_len - 1)

    # Create a mask for the choice tokens for each item in the batch
    loss_mask = torch.zeros_like(shift_labels, dtype=torch.float32, device=device)  # Use float for summing losses

    total_choice_tokens_per_item = []  # Keep track for division, handling zero case

    for i in range(batch_size):
        start_idx = choice_start_indices[i]
        length = choice_lengths[i]
        # The choice spans from token `start_idx` to `start_idx + length - 1` in the *original* input_ids.
        # When shifted, we predict tokens from `start_idx` up to `start_idx + length - 1`.
        # The corresponding indices in the `shift_labels` (length seq_len-1) are
        # from `start_idx` (predicting token `start_idx+1`)
        # up to `start_idx + length - 2` (predicting token `start_idx + length -1`).
        # However, the loss is calculated based on predicting the *next* token.
        # Logit at index `t` predicts token `t+1`. Label at index `t` is token `t+1`.
        # So, we need the losses corresponding to predicting the choice tokens.
        # The first choice token is at `input_ids[i, start_idx]`. This is `shift_labels[i, start_idx-1]` if start_idx > 0.
        # The last choice token is at `input_ids[i, start_idx + length - 1]`. This is `shift_labels[i, start_idx + length - 2]`.
        # The logits used are `shift_logits[i, start_idx-1]` to `shift_logits[i, start_idx + length - 2]`.

        # Let's rethink the mask based on `calculate_span_loss` logic:
        # Original code used `loss_mask[effective_choice_start_shifted:effective_choice_end_shifted] = True`
        # `effective_choice_start_shifted = max(0, choice_start_token_index - 1)`
        # `effective_choice_end_shifted = flat_labels.size(0)` (This seems wrong, should be end of choice)
        # The original `calculate_span_loss` mask included *all* tokens from the start of the choice onwards.
        # Let's stick to that logic for consistency, assuming it was intended.

        # effective_choice_start_shifted is the index in shift_labels/loss_per_token
        # corresponding to the prediction of the *first* choice token.
        effective_start_idx_shifted = max(0, start_idx)  # Index in the `seq_len-1` dimension

        # The original logic masked *to the end*. Let's assume it should mask *only* choice tokens.
        # Choice tokens are input_ids[start_idx] to input_ids[start_idx + length - 1]
        # These are labels shift_labels[start_idx-1] to shift_labels[start_idx + length - 2] (if start_idx > 0)
        # Or shift_labels[0] to shift_labels[length-1] (if start_idx == 0)

        # Let's mask the positions in `loss_per_token` that correspond to predicting choice tokens.
        # We predict token `t+1` using logit `t` and compare with label `t`.
        # Predict choice token 1 (at index `start_idx`): Use logit `start_idx-1`, compare with label `start_idx-1` (which is `input_ids[start_idx]`). Index `start_idx-1` in `loss_per_token`.
        # Predict choice token L (at index `start_idx+L-1`): Use logit `start_idx+L-2`, compare with label `start_idx+L-2` (which is `input_ids[start_idx+L-1]`). Index `start_idx+L-2` in `loss_per_token`.

        mask_start = max(0, start_idx - 1)  # Start index in loss_per_token (seq_len-1 dim)
        mask_end = start_idx + length - 1  # End index (exclusive) in loss_per_token
        mask_end = min(mask_end, seq_len - 1)  # Ensure it doesn't exceed bounds

        num_choice_tokens_in_loss = 0
        if mask_start < mask_end:
            loss_mask[i, mask_start:mask_end] = 1.0
            num_choice_tokens_in_loss = mask_end - mask_start  # Actual number of terms contributing to loss

        # Store the count for averaging. Use the original intended number of choice tokens for averaging.
        # The original code averaged over `num_choice_tokens = loss_mask.sum().item()`,
        # which counted tokens *after* shifting and masking. Let's replicate that.
        # The number of tokens to average over is `length` if the choice fits entirely.
        actual_num_tokens_for_avg = num_choice_tokens_in_loss  # Use the count of actual loss terms calculated

        total_choice_tokens_per_item.append(float(actual_num_tokens_for_avg))

    # Apply mask and sum loss per item
    masked_loss_per_token = loss_per_token * loss_mask  # Element-wise multiplication
    summed_loss_per_item = masked_loss_per_token.sum(dim=1)  # Sum across seq_len dimension -> [batch_size]

    # Calculate average loss per choice token for each item
    avg_loss_per_item = torch.zeros_like(summed_loss_per_item)
    num_tokens_tensor = torch.tensor(total_choice_tokens_per_item, device=device, dtype=torch.float32)

    # Avoid division by zero
    valid_mask = num_tokens_tensor > 0
    avg_loss_per_item[valid_mask] = summed_loss_per_item[valid_mask] / num_tokens_tensor[valid_mask]
    # Set loss to infinity for items with zero valid choice tokens
    avg_loss_per_item[~valid_mask] = float("inf")

    return avg_loss_per_item


# --- Evaluation Function ---


def evaluate_hellaswag(
    model: nn.Module,
    tokenizer: Tokenizer,
    loss_fn_eval: nn.Module,  # IMPORTANT: Ensure this has reduction='none'
    device: str,
    max_samples: Optional[int] = None,
    model_seq_len: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluates HellaSwag using zero-shot likelihood with batched choices."""
    benchmark_name = "HellaSwag"
    print(f"\n--- Evaluating {benchmark_name} ---")
    model.eval()

    if model_seq_len is None:
        # Try to infer from model config, fallback to default
        model_seq_len = getattr(model, "seq_len", getattr(model, "config", {}).get("max_position_embeddings", 1024))

    # Ensure loss function has reduction='none'
    if not (hasattr(loss_fn_eval, "reduction") and loss_fn_eval.reduction == "none"):
        raise ValueError("loss_fn_eval must have reduction='none' for batched span loss calculation.")

    pad_token_id = tokenizer.token_to_id("[PAD]")

    try:
        dataset = load_dataset("hellaswag", split="validation", trust_remote_code=True)
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    except Exception as e:
        print(f"Failed to load dataset hellaswag: {e}")
        return {"accuracy": 0.0, "evaluated_samples": 0}

    correct_predictions = 0
    total_evaluated = 0

    for item in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        context = item["ctx"]
        choices_text = item["endings"]  # List of strings
        try:
            true_label_idx = int(item["label"])  # Label is "0", "1", ...
            if not (0 <= true_label_idx < len(choices_text)):
                print(f"Warning: Invalid label index {true_label_idx} for item. Skipping.")
                continue  # Invalid label index
        except (ValueError, TypeError):
            print(f"Warning: Invalid label format '{item['label']}' for item. Skipping.")
            continue  # Skip if label is invalid

        batch_prepared_data = []  # List of tuples: (token_ids, choice_start, choice_len)
        max_len_in_batch = 0

        # 1. Prepare all choices for the current context
        for choice_text in choices_text:
            prepared = prepare_tokens_and_choice_start(tokenizer, context, choice_text, model_seq_len)
            if prepared is None:
                # Handle invalid choice (e.g., too long) - assign infinite loss later
                batch_prepared_data.append(None)
            else:
                token_ids, choice_start, choice_len = prepared
                batch_prepared_data.append((token_ids, choice_start, choice_len))
                max_len_in_batch = max(max_len_in_batch, len(token_ids))

        if max_len_in_batch == 0:  # All choices were invalid
            total_evaluated += 1  # Count as evaluated, but prediction will be wrong/skipped
            continue

        batch_input_ids = []
        batch_choice_starts = []
        batch_choice_lengths = []
        valid_choice_indices = []  # Track indices of valid choices in the original list

        # 2. Pad and collect valid choices into tensors
        for i, prepared_data in enumerate(batch_prepared_data):
            if prepared_data is None:
                continue  # Skip invalid choices processed earlier

            token_ids, choice_start, choice_len = prepared_data
            padding_needed = max_len_in_batch - len(token_ids)
            padded_token_ids = token_ids + ([pad_token_id] * padding_needed)

            batch_input_ids.append(padded_token_ids)
            batch_choice_starts.append(choice_start)
            batch_choice_lengths.append(choice_len)
            valid_choice_indices.append(i)

        if not batch_input_ids:  # No valid choices after padding/prep
            total_evaluated += 1
            continue

        # 3. Convert to tensors and move to device
        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)

        # 4. Run single batched inference step
        try:
            # Use autocast wrapper if not handled by model/inference_step
            with torch.no_grad(), autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(input_ids_tensor, True)

        except Exception as e:
            print(f"\nError during inference: {e}")
            # Handle potential OOM or other runtime errors
            # Option: skip sample, or try smaller batch (not applicable here)
            total_evaluated += 1  # Count as evaluated but failed
            continue  # Skip to next sample

        # 5. Calculate loss for all choices in the batch
        # Ensure lists passed to calculate_batch_span_loss match the tensors in the batch
        batch_losses = calculate_batch_span_loss(
            logits,
            input_ids_tensor,
            batch_choice_starts,  # List for the items *in the batch*
            batch_choice_lengths,  # List for the items *in the batch*
            loss_fn_eval,
            device,
        )  # Returns a tensor of shape [batch_size]

        # 6. Map losses back to original choices and find the best one
        all_choice_losses = [float("inf")] * len(choices_text)
        for i, original_idx in enumerate(valid_choice_indices):
            all_choice_losses[original_idx] = batch_losses[i].item()

        min_loss = float("inf")
        predicted_idx = -1
        for i, loss in enumerate(all_choice_losses):
            if loss < min_loss:
                min_loss = loss
                predicted_idx = i

        # 7. Update accuracy
        if predicted_idx != -1 and predicted_idx == true_label_idx:
            correct_predictions += 1
        total_evaluated += 1

    accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0
    print(f"--- Finished {benchmark_name}: Accuracy: {accuracy:.4f} ({correct_predictions}/{total_evaluated}) ---")
    model.train()  # Set model back to training mode
    return {"accuracy": accuracy, "evaluated_samples": total_evaluated}
