import torch
import torch.nn as nn
from torch.amp import autocast
from tokenizers import Tokenizer
from typing import Optional, List, Tuple

# --- Core Likelihood Calculation Helpers ---


def calculate_span_loss(
    model: nn.Module,
    input_ids: torch.Tensor,  # Shape: [1, seq_len]
    choice_start_token_index: int,
    loss_fn_eval: nn.Module,
    seq_len: int,
    device: str,
) -> float:
    """
    Calculates the average cross-entropy loss ONLY for the choice part of input_ids.
    Assumes input_ids is already padded/truncated to seq_len.
    """

    if choice_start_token_index >= seq_len - 1:
        return float("inf")

    with torch.no_grad(), autocast(device_type="cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16):
        logits = inference_step(model, input_ids)  # Get vocab logits

    shift_logits = logits[..., :-1, :].contiguous()  # [1, seq_len-1, vocab_size]
    shift_labels = input_ids[..., 1:].contiguous()  # [1, seq_len-1]

    flat_logits = shift_logits.view(-1, shift_logits.size(-1))  # [seq_len-1, vocab_size]
    flat_labels = shift_labels.view(-1)  # [seq_len-1]

    loss_mask = torch.zeros_like(flat_labels, dtype=torch.bool)
    effective_choice_start_shifted = max(0, choice_start_token_index - 1)
    effective_choice_end_shifted = flat_labels.size(0)

    if effective_choice_start_shifted < effective_choice_end_shifted:
        loss_mask[effective_choice_start_shifted:effective_choice_end_shifted] = True

    num_choice_tokens = loss_mask.sum().item()
    if num_choice_tokens == 0:
        return float("inf")

    masked_labels = torch.where(loss_mask, flat_labels, torch.tensor(loss_fn_eval.ignore_index).to(device))
    # Calculate element-wise loss (assuming reduction='none')
    elementwise_loss = loss_fn_eval(flat_logits, masked_labels)  # Shape [seq_len-1]
    # Sum the losses. Since ignored indices produce 0 loss, this effectively sums choice token losses.
    sum_choice_loss = elementwise_loss.sum()
    # Divide by the number of actual choice tokens to get the average loss for the choice span.
    average_loss_per_choice_token = sum_choice_loss / num_choice_tokens

    return average_loss_per_choice_token.item()


def prepare_input_and_find_choice_start(
    tokenizer: Tokenizer, context: str, choice: str, max_seq_len: int
) -> Optional[Tuple[List[int], int]]:
    """
    Tokenizes context and choice, combines, truncates, pads and returns token IDs
    and the start index of the choice tokens. Returns None if invalid.
    """
    choice_prefix = " "  # Standard prefix
    tokenized_context = tokenizer.encode(context).ids
    # Tokenize choice with prefix to potentially handle leading space meaning
    tokenized_choice = tokenizer.encode(choice_prefix + choice).ids

    full_token_ids = tokenized_context + tokenized_choice
    choice_start_index = len(tokenized_context)

    # Truncation (prioritize keeping the choice)
    if len(full_token_ids) > max_seq_len:
        choice_len = len(tokenized_choice)
        keep_context_len = max_seq_len - choice_len
        if keep_context_len < 0:
            return None  # Choice too long
        truncated_context_tokens = tokenized_context[-keep_context_len:]
        full_token_ids = truncated_context_tokens + tokenized_choice
        choice_start_index = len(truncated_context_tokens)

    # Padding
    pad_token_id = tokenizer.token_to_id("[PAD]")
    if pad_token_id is None:
        pad_token_id = 0  # Fallback

    padding_needed = max_seq_len - len(full_token_ids)
    if padding_needed > 0:
        full_token_ids = full_token_ids + ([pad_token_id] * padding_needed)
    elif padding_needed < 0:
        full_token_ids = full_token_ids[:max_seq_len]  # Should not happen

    if choice_start_index >= max_seq_len:
        return None  # Choice effectively not included

    return full_token_ids, choice_start_index


def inference_step(model: nn.Module, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper around model inference to ensure consistent interface.
    Returns: (logits, pre_output)
    """
    model_output, _ = model(x, True)
    return model_output
