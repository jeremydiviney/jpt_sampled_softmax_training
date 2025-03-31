import torch
import torch.nn as nn
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Optional

from .utils import prepare_input_and_find_choice_start, calculate_span_loss


def evaluate_arc_easy(
    model: nn.Module,
    tokenizer: Tokenizer,
    loss_fn_eval: nn.Module,
    max_samples: Optional[int] = None,
    model_seq_len: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluates ARC-Easy using zero-shot likelihood."""
    benchmark_name = "ARC-Easy"
    print(f"\n--- Evaluating {benchmark_name} ---")
    model.eval()

    device = model.device

    if model_seq_len is None:
        model_seq_len = getattr(model, "seq_len", 1024)

    try:
        dataset = load_dataset("ai2_arc", "ARC-Easy", split="validation", trust_remote_code=True)
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    except Exception as e:
        print(f"Failed to load dataset ai2_arc/ARC-Easy: {e}")
        return {"accuracy": 0.0, "evaluated_samples": 0}

    correct_predictions = 0
    total_evaluated = 0

    for item in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        context = item["question"]
        choices_data = item["choices"]  # Dict with 'text' and 'label' lists
        choices = choices_data["text"]
        choice_labels = choices_data["label"]  # e.g., ["A", "B", "C", "D"] or ["1", "2", "3", "4"]

        raw_label = item["answerKey"]
        try:
            # Find the index corresponding to the answer key label
            true_label_idx = choice_labels.index(raw_label)
            if not (0 <= true_label_idx < len(choices)):
                continue
        except ValueError:
            continue  # Skip if answerKey is not in choice_labels

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
    return {"accuracy": accuracy, "evaluated_samples": total_evaluated}
