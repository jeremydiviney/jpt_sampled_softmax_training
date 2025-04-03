import torch
import torch.nn as nn
from torch.amp import autocast
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Optional, List, Tuple  # Keep necessary types

# --- Helper Functions ---


def prepare_tokens_and_span_info(tokenizer: Tokenizer, context: str, choice: str, max_seq_len: int) -> Optional[Tuple[List[int], int, int]]:
    """
    Tokenizes context and choice, combines, truncates prioritizing choice.
    Returns UNPADDED token IDs, the start index of choice tokens,
    and the number of choice tokens. Returns None if invalid.
    (Adapted from the first version you provided)
    """
    choice_prefix = " "  # Ensure choice starts as a new "word"
    tokenized_context = tokenizer.encode(context).ids
    tokenized_choice = tokenizer.encode(choice_prefix + choice, add_special_tokens=False).ids
    choice_len = len(tokenized_choice)

    print(f"Context tokens: {len(tokenized_context)}, Choice tokens: {choice_len}")
    print(f"Context: '{context[:30]}...'")
    print(f"Choice: '{choice}'")

    if choice_len == 0:
        print("  Invalid: Empty choice")
        return None  # Invalid choice

    full_token_ids = tokenized_context + tokenized_choice
    choice_start_index = len(tokenized_context)

    # Truncation (prioritize keeping the choice)
    if len(full_token_ids) > max_seq_len:
        keep_context_len = max_seq_len - choice_len
        if keep_context_len < 0:
            print(f"  Invalid: Choice too long ({choice_len} > {max_seq_len})")
            return None  # Choice itself is too long

        # Truncate context from the left
        truncated_context_tokens = tokenized_context[-keep_context_len:]
        full_token_ids = truncated_context_tokens + tokenized_choice
        choice_start_index = len(truncated_context_tokens)  # Recalculate start index
        print(f"  Truncated: Context from {len(tokenized_context)} to {len(truncated_context_tokens)} tokens")

    # FIXED: Ensure the choice is not at the very end of the sequence
    # If the choice is at the end, we need to add a dummy token after it
    # so that we can calculate loss for the last token of the choice
    if choice_start_index + choice_len == len(full_token_ids):
        # Add a dummy token (e.g., the first token of the context)
        if len(tokenized_context) > 0:
            dummy_token = tokenized_context[0]
            full_token_ids.append(dummy_token)
            print(f"  Added dummy token at the end to allow loss calculation for the last choice token")
        else:
            # If we don't have any context tokens, use a special token
            # This is a fallback and should rarely happen
            dummy_token = tokenizer.token_to_id("[SEP]") if tokenizer.token_to_id("[SEP]") is not None else 0
            full_token_ids.append(dummy_token)
            print(f"  Added special token at the end to allow loss calculation for the last choice token")

    # Final check
    if choice_start_index >= len(full_token_ids) and choice_len > 0:
        print(f"  Invalid: Choice start index ({choice_start_index}) >= full length ({len(full_token_ids)})")
        return None

    print(f"  Valid: Total tokens={len(full_token_ids)}, Choice start={choice_start_index}, Choice length={choice_len}")
    return full_token_ids, choice_start_index, choice_len


def calculate_batch_span_loss(
    logits: torch.Tensor,  # Shape: [batch_size, seq_len, vocab_size]
    input_ids: torch.Tensor,  # Shape: [batch_size, seq_len]
    choice_start_indices: List[int],  # List of start indices, len = batch_size
    choice_lengths: List[int],  # List of choice lengths, len = batch_size
    loss_fn_eval: nn.Module,  # Expects reduction='none'
) -> torch.Tensor:  # Shape: [batch_size] - loss per item
    """
    Calculates the average cross-entropy loss ONLY for the choice part
    for each item in the batch.
    (Adapted from the first version you provided)
    """
    batch_size, seq_len, _ = logits.shape
    device = logits.device

    # Debug info
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"Choice start indices: {choice_start_indices}")
    print(f"Choice lengths: {choice_lengths}")

    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()  # Shape: [batch_size, seq_len-1]

    # Calculate element-wise loss
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    elementwise_loss = loss_fn_eval(flat_logits, flat_labels)  # Shape: [batch_size * (seq_len-1)]

    # Reshape loss back to batch dimension
    loss_per_token = elementwise_loss.view(batch_size, seq_len - 1)  # Shape: [batch_size, seq_len-1]

    # Create mask for the choice tokens
    loss_mask = torch.zeros_like(loss_per_token, dtype=torch.float32)
    num_tokens_per_item = torch.zeros(batch_size, device=device, dtype=torch.float32)

    for i in range(batch_size):
        start_idx = choice_start_indices[i]
        length = choice_lengths[i]

        # Debug info
        # print(f"Item {i}: start_idx={start_idx}, length={length}")

        # FIXED: Handle the case where the choice is at the end of the sequence
        # For a choice at position N with length L, we need to predict tokens at positions N+1 to N+L
        # But since we're using shifted logits/labels, we need to adjust:
        # - For position N+1, we use logits at position N, labels at position N+1
        # - For position N+L, we use logits at position N+L-1, labels at position N+L

        # Calculate the range of positions in the shifted sequence where we need to calculate loss
        # For a choice at position N with length L, we need positions N to N+L-1 in the shifted sequence
        mask_start = start_idx
        mask_end = start_idx + length - 1  # Changed from length to length-1

        # Clip indices to be valid for loss_per_token (shape [batch_size, seq_len-1])
        mask_start = max(0, mask_start)
        mask_end = min(seq_len - 2, mask_end)  # Max index is seq_len - 2 (since we shifted)

        # print(f"  Mask range: [{mask_start}, {mask_end+1})")  # +1 for display purposes

        if mask_start <= mask_end:  # Changed from < to <= to handle single-token choices
            loss_mask[i, mask_start : mask_end + 1] = 1.0  # +1 to make the end inclusive
            num_tokens_per_item[i] = float(mask_end - mask_start + 1)  # +1 to count inclusive
            print(f"  Valid mask: {mask_end - mask_start + 1} tokens")
        else:
            print(f"  Invalid mask: mask_start > mask_end")

    # Debug info
    # print(f"Number of tokens per item: {num_tokens_per_item}")
    # print(f"Valid items: {torch.sum(num_tokens_per_item > 0).item()} out of {batch_size}")

    # Apply mask and sum loss per item
    masked_loss = loss_per_token * loss_mask
    summed_loss = masked_loss.sum(dim=1)  # Shape: [batch_size]

    # Calculate average loss per choice token for each item
    # Avoid division by zero, set loss to infinity for invalid spans
    avg_loss = torch.full_like(summed_loss, float("inf"))
    valid_mask = num_tokens_per_item > 0
    avg_loss[valid_mask] = summed_loss[valid_mask] / num_tokens_per_item[valid_mask]

    # Debug info
    # print(f"Final losses: {avg_loss}")
    # print(f"Number of infinite losses: {torch.sum(torch.isinf(avg_loss)).item()} out of {batch_size}")

    return avg_loss


def evaluate_winogrande(
    model: nn.Module,
    tokenizer: Tokenizer,
    loss_fn_eval: nn.Module,  # IMPORTANT: Ensure this has reduction='none'
    max_samples: Optional[int] = None,
    model_seq_len: Optional[int] = None,
    winogrande_config: str = "winogrande_xl",  # Common config
) -> Dict[str, float]:
    """Evaluates Winogrande (XL) using zero-shot likelihood with batched choices."""
    benchmark_name = f"Winogrande_{winogrande_config}"
    print(f"\n--- Evaluating {benchmark_name} ---")
    model.eval()

    # Determine device and sequence length
    try:
        device = next(model.parameters()).device
    except Exception:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model_seq_len is None:
        model_seq_len = getattr(model.config, "max_position_embeddings", 1024)

    print(f"Using device: {device}, model_seq_len: {model_seq_len}")

    # Validate loss function
    if not (hasattr(loss_fn_eval, "reduction") and loss_fn_eval.reduction == "none"):
        raise ValueError("loss_fn_eval must have reduction='none'")

    # Get padding token ID
    pad_token_id = tokenizer.token_to_id("[PAD]")
    print(f"Padding token ID: {pad_token_id}")

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
    skipped_items = 0
    invalid_choices = 0

    for item_idx, item in enumerate(tqdm(dataset, desc=f"Evaluating {benchmark_name}")):
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]
        answer = item["answer"]  # "1" or "2"

        # Extract context
        context_parts = sentence.split("_", 1)
        if len(context_parts) != 2:
            skipped_items += 1
            continue  # Skip malformed items
        context = context_parts[0].strip()

        # Prepare choices
        choices_text = [option1, option2]
        prepared_choices = []  # Stores (ids, start, length) or None
        max_len = 0
        for i, choice in enumerate(choices_text):
            prep = prepare_tokens_and_span_info(tokenizer, context, choice, model_seq_len)
            prepared_choices.append(prep)
            if prep:
                max_len = max(max_len, len(prep[0]))
                print(f"Item {item_idx}, Choice {i}: Valid preparation with {len(prep[0])} tokens, start={prep[1]}, length={prep[2]}")
            else:
                print(f"Item {item_idx}, Choice {i}: Invalid preparation")
                invalid_choices += 1

        # Filter out invalid preparations and check if any choice is valid
        valid_preps = [(i, prep) for i, prep in enumerate(prepared_choices) if prep is not None]
        if not valid_preps or max_len == 0:
            skipped_items += 1
            continue  # Cannot proceed if no choice is valid

        # Pad and batch valid choices
        batch_input_ids = []
        batch_choice_starts = []
        batch_choice_lengths = []
        original_indices = []  # Track which original choice (0 or 1) this corresponds to

        for original_idx, (token_ids, start, length) in valid_preps:
            padding = [pad_token_id] * (max_len - len(token_ids))
            batch_input_ids.append(token_ids + padding)
            batch_choice_starts.append(start)
            batch_choice_lengths.append(length)
            original_indices.append(original_idx)

        input_ids_tensor = torch.tensor(batch_input_ids, dtype=torch.long, device=device)
        print(f"Batch shape: {input_ids_tensor.shape}")

        # Run Inference
        with torch.no_grad(), autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, _ = model(input_ids_tensor, True)
            print(f"Logits shape: {logits.shape}")

            # Calculate Losses
        batch_losses = calculate_batch_span_loss(
            logits, input_ids_tensor, batch_choice_starts, batch_choice_lengths, loss_fn_eval
        )  # Tensor of losses for items in the batch

        # Map losses back and find best choice
        all_choice_losses = [float("inf")] * len(choices_text)
        for i, loss in enumerate(batch_losses):
            all_choice_losses[original_indices[i]] = loss.item()  # Map back to original index (0 or 1)

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
                print(f"Warning: Invalid label '{answer}' encountered during checking.")
                # Don't count as correct, but still evaluated.
            total_evaluated += 1  # Increment only if a valid prediction was possible

    accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0.0
    print(f"--- Finished {benchmark_name}: Accuracy: {accuracy:.4f} ({correct_predictions}/{total_evaluated}) ---")
    print(f"Skipped items: {skipped_items}, Invalid choices: {invalid_choices}")
    model.train()  # Set model back to training mode
    return {"accuracy": accuracy, "evaluated_samples": total_evaluated}
