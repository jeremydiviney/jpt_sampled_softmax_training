import torch
import torch.nn as nn
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, Optional

from .utils import prepare_input_and_find_choice_start, calculate_span_loss


def evaluate_winogrande(
    model: nn.Module,
    tokenizer: Tokenizer,
    loss_fn_eval: nn.Module,
    max_samples: Optional[int] = None,
    model_seq_len: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluates Winogrande (XL) using zero-shot likelihood."""
    benchmark_name = "Winogrande"
    print(f"\n--- Evaluating {benchmark_name} ---")
    model.eval()

    device = model.device

    if model_seq_len is None:
        model_seq_len = getattr(model, "seq_len", 1024)

    try:
        # Using winogrande_xl config
        dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
        if max_samples is not None and max_samples > 0:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
    except Exception as e:
        print(f"Failed to load dataset winogrande/winogrande_xl: {e}")
        return {"accuracy": 0.0, "evaluated_samples": 0}

    correct_predictions = 0
    total_evaluated = 0

    for item in tqdm(dataset, desc=f"Evaluating {benchmark_name}"):
        sentence = item["sentence"]  # Sentence with placeholder "_"
        option1 = item["option1"]
        option2 = item["option2"]
        choices = [option1, option2]

        try:
            # Label is "1" or "2", map to 0 or 1
            true_label_idx = int(item["answer"]) - 1
            if not (0 <= true_label_idx < len(choices)):
                continue
        except (ValueError, TypeError):
            continue

        choice_losses = []
        min_loss = float("inf")
        predicted_idx = -1

        for i, choice_text in enumerate(choices):
            # Format the sentence by replacing the placeholder
            full_text = sentence.replace("_", choice_text)
            # For span loss, context is tricky. Let's try context = sentence up to placeholder?
            # This is an approximation. A simpler alternative is to calculate loss over the *whole* formatted text.
            # Let's stick to the "completion" idea: context is the part before the choice text appears.
            context = sentence.split("_")[0]  # Approximate context
            eval_choice = choice_text  # The text filling the blank

            prepared = prepare_input_and_find_choice_start(tokenizer, context, eval_choice, model_seq_len)

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
