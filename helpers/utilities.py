import numpy as np


def calculate_text_accuracy(pred_texts, target_texts, pad_token_char):
    pred_chars = np.array(pred_texts)
    target_chars = np.array(target_texts)
    # Create mask to ignore pad tokens
    mask = (target_chars == pad_token_char) | (target_chars == "")

    # Apply mask for counting
    valid_chars = np.sum(~mask).item()
    valid_tokens = np.sum(np.any(~mask, axis=2)).item()

    adjusted_pred_chars = pred_chars.copy()
    adjusted_target_chars = target_chars.copy()

    # set all pad tokens to "" to make the match comparison easier
    adjusted_pred_chars[mask] = ""
    adjusted_target_chars[mask] = ""

    # Count matches only for non-pad positions
    char_matches = np.sum((adjusted_pred_chars == target_chars) & ~mask).item()
    token_matches = np.sum(np.all((adjusted_pred_chars == target_chars), axis=2)).item()

    return {
        "char_count": valid_chars,
        "token_count": valid_tokens,
        "char_matches": char_matches,
        "token_matches": token_matches,
    }


def calculate_token_accuracy(pred_tokens, target_tokens, pad_token):
    pred_tokens = np.array(pred_tokens)
    target_tokens = np.array(target_tokens)
    # Create mask to ignore pad tokens
    mask = (target_tokens == pad_token) | (target_tokens == "")

    # Apply mask for counting
    valid_tokens = np.sum(~mask).item()

    # Count matches only for non-pad positions
    token_matches = np.sum((pred_tokens == target_tokens) & ~mask).item()

    return {
        "token_count": valid_tokens,
        "token_matches": token_matches,
    }
