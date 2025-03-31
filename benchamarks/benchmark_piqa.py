import torch
import torch.nn as nn
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Optional

from .utils import prepare_input_and_find_choice_start, calculate_span_loss


def evaluate_piqa(
    model: nn.Module,
    tokenizer: Tokenizer,
    loss_fn_eval: nn.Module,
    max_samples: Optional[int] = None,
    model_seq_len: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluates PIQA using zero-shot likelihood."""
    benchmark_name = "PIQA"
    print(f"\n--- Evaluating {benchmark_name} ---")
    model.eval()

    device = model.device

    if model_seq_len is None:
        model_seq_len = getattr(model, "seq_len", 1024)

    try:
        dataset = load_dataset("piqa", split="validation", trust_remote_code=True)
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    except Exception as e:
        print(f"Failed to load dataset piqa: {e}")
        return {"accuracy": 0.0, "evaluated_samples": 0}

    correct_predictions = 0
    total_evaluated = 0

    for item in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        context = item["goal"]  # The goal acts as context
        choice1 = item["sol1"]
        choice2 = item["sol2"]
        choices = [choice1, choice2]

        try:
            true_label_idx = int(item["label"])  # Label is 0 or 1
            if not (0 <= true_label_idx < len(choices)):
                continue
        except (ValueError, TypeError):
            continue

        choice_losses = []
        min_loss = float("inf")
        predicted_idx = -1

        for i, choice_text in enumerate(choices):
            prepared = prepare_input_and_find_choice_start(tokenizer, context, choice_text, model_seq_len)
            if prepared is None:
                choice_losses.append(float("inf"))
                continue

            input_ids_list, choice_start_idx = prepared
            input_tensor = torch.tensor([input_ids_list], device=device)

            loss = calculate_span_loss(model, input_tensor, choice_start_idx, loss_fn_eval, model_seq_len)
            choice_losses.append(loss)

            if loss < min_loss:
                min_loss = loss
                predicted_idx = i

        if predicted_idx != -1 and predicted_idx == true_label_idx:
            correct_predictions += 1
        total_evaluated += 1

    accuracy = correct_predictions / total_evaluated if total_evaluated > 0 else 0
    print(f"--- Finished {benchmark_name}: Accuracy: {accuracy:.4f} ({correct_predictions}/{total_evaluated}) ---")
    model.train()
    return {f"{benchmark_name}_accuracy": accuracy, f"{benchmark_name}_evaluated_samples": total_evaluated}
