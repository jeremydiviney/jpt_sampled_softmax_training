import os
from typing import Optional
from datetime import datetime
import time
import sys
import inspect

import math

from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from tokenizers import Tokenizer

from sklearn.neighbors import KDTree

from models.jpt1 import JPT1

from datasources.fineweb10B import get_or_train_tokenizer, Fineweb10BDataset
from helpers.experiments import run_experiment, count_parameters, create_experiments
from helpers.training import (
    save_model,
    enable_torch_optimizations,
    setup_flash_attention,
)

# Import distributed utilities
from helpers.distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    get_model_for_training,
    is_main_process,
    get_world_size,
    reduce_value,
)


from datasources.fineweb10B import load_hf_dataset, Fineweb10BDataset

from models.jpt1 import JPT1ModelType


from helpers.utilities import calculate_token_accuracy


# --------------------------------------------------
# 2. Model Definition
# --------------------------------------------------


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    loss_fn: torch.nn.Module,
    local_rank: int,
    distributed: bool,
) -> dict:

    assert local_rank == 0, "Evaluation must be done on main process"

    model.eval()

    raw_model = model.module if distributed else model

    dataset = dataloader.dataset
    tokenizer = dataset.tokenizer

    total_loss = 0
    total_loss_norm = 0
    batch_count = 0

    token_matches_total = 0
    token_total = 0

    norm_loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

    do_final_projection = raw_model.model_type != JPT1QuantModelType.STANDARD_SAMPLED

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            jpt_output, loss = inference_and_loss_step(model, x, y, loss_fn, do_final_projection, distributed)

            jpt_output_norm, loss_norm = inference_and_loss_step(model, x, y, norm_loss_fn, True, distributed)

            total_loss += loss.item()
            total_loss_norm += loss_norm.item()

            batch_count += 1

            pred_token_indices = jpt_output_norm.argmax(dim=-1)

            pred_token_indices = pred_token_indices.detach().cpu().numpy()

        # Access the base model for token generation functions

        pred_tokens = raw_model.get_text_token_from_indices(pred_token_indices)
        target_tokens = raw_model.get_text_token_from_indices(y.detach().cpu().numpy())

        accuracy_metrics = calculate_token_accuracy(pred_tokens, target_tokens, dataset.token_list["[PAD]"])

        token_matches_total += accuracy_metrics["token_matches"]
        token_total += accuracy_metrics["token_count"]

        print(f"\nSample {batch_count}:")
        print(f"Current token accuracy: {token_matches_total/token_total:.2%}")

    print("Generating text...")
    # Generate text only on main process

    for _ in range(20):
        generate_text(raw_model, "Hello, I'm a language model,", 50, dataloader.dataset, 0.5, local_rank)

    # sync all distributed processes

    result = {
        "val_loss": total_loss / batch_count,
        "val_loss_norm": total_loss_norm / batch_count,
        "val_token_accuracy": token_matches_total / token_total,
    }

    model.train()
    return result


def inference_step(model, x, do_final_projection: bool):

    model_output, pre_output = model(x, do_final_projection)
    return model_output, pre_output


class CustomSampledLoss(torch.nn.Module):
    def __init__(self, ignore_index: int, total_compare_tokens: int):
        super().__init__()
        self.ignore_index = ignore_index
        self.total_compare_tokens = total_compare_tokens

    def forward(self, model, hidden_states, target_indices):
        batch_size, seq_length, hidden_dim = hidden_states.shape
        vocab_size = model.lookup_embeddings.weight.size(0)

        # Flatten tensors
        flat_hidden = hidden_states.reshape(-1, hidden_dim)  # [batch_size*seq_length, hidden_dim]
        flat_targets = target_indices.reshape(-1)  # [batch_size*seq_length]

        # Ignore padding tokens
        ignore_mask = flat_targets == self.ignore_index
        flat_hidden = flat_hidden[~ignore_mask]
        flat_targets = flat_targets[~ignore_mask]

        all_embeddings = model.lookup_embeddings.weight

        # Get unique targets from the batch
        unique_targets = torch.unique(flat_targets)

        # Sample additional negative indices
        # Target number of negative samples
        num_batch_uniques = unique_targets.shape[0]
        num_extra_samples = max(0, self.total_compare_tokens - num_batch_uniques)

        # Sample from non-batch tokens
        sampling_mask = torch.ones(vocab_size, device=flat_hidden.device, dtype=torch.bool)
        sampling_mask[unique_targets] = False
        valid_indices = torch.nonzero(sampling_mask, as_tuple=True)[0]

        # Get extra negative indices
        if num_extra_samples > 0 and len(valid_indices) > 0:
            perm = torch.randperm(len(valid_indices), device=valid_indices.device)
            num_to_sample = min(num_extra_samples, len(valid_indices))
            extra_neg_indices = valid_indices[perm[:num_to_sample]]
        else:
            extra_neg_indices = torch.tensor([], device=flat_hidden.device, dtype=torch.long)

        # Combine all indices for comparisons - positives first, then uniques, then extras
        all_indices = torch.cat([unique_targets, extra_neg_indices])  # Include all unique targets (which includes all positives)

        # Get embeddings for all indices
        comparison_embeddings = all_embeddings[all_indices]  # [num_comparisons, hidden_dim]

        # Compute all similarities at once
        all_similarities = torch.matmul(flat_hidden, comparison_embeddings.t())

        # Create targets tensor - map each flat_target to its position in all_indices
        # First create a mapping from token IDs to their positions in all_indices
        indices_map = torch.zeros(vocab_size, dtype=torch.long, device=flat_hidden.device)
        indices_map[all_indices] = torch.arange(len(all_indices), device=flat_hidden.device)

        # Use the mapping to get the target positions
        targets = indices_map[flat_targets]

        # Apply cross entropy loss
        loss = F.cross_entropy(all_similarities, targets)

        return loss


def inference_and_loss_step(model, x, y, loss_fn, do_final_projection: bool, distributed: bool):

    model_output, pre_output = inference_step(model, x, do_final_projection)  # [batch_size, seq_len, embed_dim]

    raw_model = model.module if distributed else model

    model_type = raw_model.model_type

    logits = model_output

    if model_type == JPT1QuantModelType.STANDARD_SAMPLED:
        if do_final_projection:
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))  # get model output logits in this case
        else:
            loss = loss_fn(raw_model, pre_output, y)  # get model output logits in this case

    else:
        # For standard model types, compute logits normally and apply cross entropy over the vocab.

        # logits shape is assumed to be [batch, seq_len, vocab_size]
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

    return logits, loss


def get_grad_accum_size(completion_percentage: float, batch_tokens: int, grad_accum_size: int, hit_max_at: float) -> int:

    max_facter = 1 / hit_max_at

    return max(batch_tokens, min(grad_accum_size, max_facter * completion_percentage * grad_accum_size))


def create_optimizer(model, lr, beta2, weight_decay=0.01):

    # Separate parameters that should have weight decay applied from those that shouldn't
    decay_params = []
    no_decay_params = []

    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optimizer_groups = [{"params": decay_params, "weight_decay": weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}]

    device_type = decay_params[0].device.type

    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"

    # Create optimizer with parameter groups
    return optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, beta2), fused=use_fused)


@contextmanager
def maybe_no_sync(model, distributed, sync_grads: bool):
    if not sync_grads and distributed:
        with model.no_sync():
            yield
    else:
        yield


def train_model(
    wandb,
    model,
    train_dataloader,
    val_dataloader,
    config: dict,
    loss_fn: torch.nn.Module,
    distributed: bool = False,
    local_rank: int = 0,
):
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Prepare model for distributed training if needed
    if distributed:
        model = get_model_for_training(model, device, distributed, local_rank)
    else:
        model = model.to(device)

    raw_model = model.module if distributed else model

    count_parameters(raw_model)

    optimizer = create_optimizer(model, config["lr"], config["beta2"], config["weight_decay"])

    seq_len = config["seq_len"]
    batch_size = config["batch_size"]

    # Adjust batch tokens for distributed training
    world_size = get_world_size(distributed)

    batch_tokens = batch_size * seq_len

    log_step_count = 0
    grad_accum_size = config["grad_accum_size"]
    log_step_size = config["log_step_size"]

    completion_percentage = 0.0

    early_end_pct = config["early_end_pct"]

    grad_accum_max_at = config["grad_accum_max_at"]

    # Adjust logging steps based on world size
    total_tokens = train_dataloader.dataset.token_count

    assert log_step_size % (grad_accum_size) == 0, "log_step_size must be divisible by grad_accum_size"
    assert log_step_size >= (grad_accum_size), "log_step_size must be greater than or equal to grad_accum_size * world_size"
    assert grad_accum_size % (batch_tokens * world_size) == 0, "grad_accum_size must be divisible by batch_size * world_size"

    logging_steps = 1 + (config["epochs"] * total_tokens) // log_step_size

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        total_steps=logging_steps,
        pct_start=config["warmup_pct"],
        anneal_strategy="cos",
        cycle_momentum=False,
        div_factor=10,
        final_div_factor=1,
    )

    dataset = train_dataloader.dataset

    current_lr = config["lr"]
    low_loss = 10e10

    train_time_start = time.time()

    loss_history = []

    tokens_since_step = 0
    tokens_since_grad_accum = 0

    do_final_projection = raw_model.model_type != JPT1QuantModelType.STANDARD_SAMPLED

    for epoch in range(config["epochs"]):
        batch_count = 0

        loss_accum = 0

        train_step_start = time.time()

        # Reset sampler for distributed training
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)

        current_grad_accum_step_count = 0

        for x, y in train_dataloader:
            x = x.to(device)
            y = y.to(device)

            tokens_processed = x.shape[0] * x.shape[1] * world_size
            tokens_since_step += tokens_processed
            tokens_since_grad_accum += tokens_processed

            # Grad accum size is a function of the completion percentage we reach 100% at 50% completion
            current_grad_accum_size = get_grad_accum_size(
                completion_percentage, batch_tokens * world_size, grad_accum_size, grad_accum_max_at
            )

            grad_accum_step_count = math.ceil(current_grad_accum_size / (batch_tokens * world_size))

            if is_main_process(distributed, local_rank) and batch_count % 100 == 0:
                print(f"Current grad accum size: {current_grad_accum_size}")
                print(f"Grad accum step count: {grad_accum_step_count}")

            batch_count += 1

            sync_grads = distributed and current_grad_accum_step_count == (grad_accum_step_count - 1)

            with maybe_no_sync(model, distributed, sync_grads):
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    jpt_output, loss = inference_and_loss_step(model, x, y, loss_fn, do_final_projection, distributed)

                loss = loss / grad_accum_step_count
                loss_accum += loss.detach()

                loss.backward()

            current_grad_accum_step_count += 1

            if tokens_since_grad_accum >= current_grad_accum_size:

                # print(f"Rank: {local_rank},batch_count: {batch_count}, accum step complete")

                current_grad_accum_step_count = 0
                # Add gradient clipping
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

                torch.cuda.synchronize()

                current_loss = loss_accum

                # Reduce loss across all processes if distributed
                if distributed:
                    current_loss = reduce_value(loss_accum, average=True).item()

                loss_history.append(current_loss)
                if len(loss_history) > 50:
                    loss_history.pop(0)

                current_mean_loss = sum(loss_history) / len(loss_history)

                loss_accum = 0

                # Calculate tokens per second (accounting for all processes)
                tokens_per_second = tokens_since_grad_accum / (time.time() - train_step_start)

                tokens_since_grad_accum = 0

                if current_mean_loss < low_loss:
                    low_loss = current_mean_loss
                    if is_main_process(distributed, local_rank):
                        print(
                            f"\nNew low loss: {low_loss:.7f}, Batch Time: {time.time() - train_step_start:.2f},Grad Norm: {norm:.2f}, Tokens per second: {tokens_per_second:.2f}"
                        )

                if tokens_since_step >= log_step_size:
                    tokens_since_step = 0
                    log_step_count += 1

                    if is_main_process(distributed, local_rank):
                        wandb.log(
                            {
                                "loss": current_mean_loss,
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "tokens_per_second": tokens_per_second,
                                "current_grad_accum_size": current_grad_accum_size,
                                "grad_norm": norm,
                            }
                        )

                        if log_step_count % 100 == 0:
                            eval_results = evaluate_model(model, val_dataloader, device, loss_fn, local_rank, distributed)
                            val_loss = eval_results["val_loss"]
                            val_loss_norm = eval_results["val_loss_norm"]
                            val_token_accuracy = eval_results["val_token_accuracy"]
                            wandb.log(
                                {
                                    "val_loss": val_loss,
                                    "val_loss_norm": val_loss_norm,
                                    "val_token_accuracy": val_token_accuracy,
                                    "epoch": epoch,
                                }
                            )
                            print(
                                f"\nEpoch {epoch} train_loss: {current_mean_loss:.4f}, val_loss: {val_loss:.4f}, val_loss_norm: {val_loss_norm:.4f}, "
                                f"val_token_accuracy: {val_token_accuracy:.2%}"
                                f"tokens_per_second: {tokens_per_second:.2f}"
                            )

                    completion_percentage = log_step_count / logging_steps
                    scheduler.step()

                    if distributed:
                        torch.distributed.barrier()

                torch.cuda.synchronize()
                train_step_start = time.time()

                if early_end_pct is not None and completion_percentage > early_end_pct:
                    break

                # if is_main_process(distributed, local_rank):
                # time.sleep(2)

                # if distributed:
                #     torch.distributed.barrier()

    print("Training completed!")
    # Final Evaluation - only on main process

    if distributed:
        torch.distributed.barrier()

    if is_main_process(distributed, local_rank):

        eval_results = evaluate_model(model, val_dataloader, device, loss_fn, local_rank, distributed)

        wandb.log(
            {
                "val_loss": eval_results["val_loss"],
                "val_loss_norm": eval_results["val_loss_norm"],
                "val_token_accuracy": eval_results["val_token_accuracy"],
                "epoch": epoch,
            }
        )

        train_time_end = time.time()
        print(f"Training time: {train_time_end - train_time_start:.4f} seconds")

        # Save both models
        save_dir = "saved_models"
        timestamp = datetime.now().isoformat()
        model_name = f"jpt1_{timestamp}"

        # Get model without DDP wrapper for saving
        model_to_save = model.module if distributed else model
        save_model(model_to_save, save_dir, f"{model_name}_jpt1")

    if distributed:
        torch.distributed.barrier()

    return model


def load_model(
    model: any,
    load_dir: str,
    model_name: str,
    device: Optional[str] = None,
    encoder_only: bool = False,
    decoder_only: bool = False,
) -> nn.Module:
    """
    Load a saved model state.

    Args:
        model: The model to load state into
        load_dir: Directory containing the saved model(s)
        model_name: Base name of the saved model files
        device: Device to load the model to
        encoder_only: Only load the encoder part
        decoder_only: Only load the decoder part

    Returns:
        The model with loaded state
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if encoder_only and decoder_only:
        raise ValueError("Cannot specify both encoder_only and decoder_only")

    def clean_state_dict(state_dict):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_orig_mod.", "")
            new_state_dict[new_key] = value
        return new_state_dict

    if encoder_only:
        encoder_path = os.path.join(load_dir, f"{model_name}_encoder.pt")
        state_dict = torch.load(encoder_path, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))
    elif decoder_only:
        decoder_path = os.path.join(load_dir, f"{model_name}_decoder.pt")
        state_dict = torch.load(decoder_path, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))
    else:
        full_model_path = os.path.join(load_dir, f"{model_name}_full.pt")
        state_dict = torch.load(full_model_path, map_location=device)
        model.load_state_dict(clean_state_dict(state_dict))

    return model.to(device)


def get_text_token_from_prediction_text(prediction_text: str) -> str:
    pred_text_chars = []
    for char in prediction_text:
        if char == "<EOT>":
            break
        pred_text_chars.append("" if char == "[PAD]" else char)

    return "".join(pred_text_chars)


def generate_text(
    model: nn.Module,
    prompt: str,
    max_new_tokens: int,
    dataset: Fineweb10BDataset,
    temperature: float = 0.5,
    local_rank: int = 0,
    device: str = "cuda",
) -> str:
    # Set models to eval mode
    model.eval()

    assert local_rank == 0, "Text generation must be done on main process"

    print(f"\n\nPrompt: {prompt}", end="", flush=True)

    if len(prompt) == 0:
        raise ValueError("Prompt must be at least one character long")

    result: [str] = [prompt]

    for _ in range(max_new_tokens):

        current_context = "".join(result)
        # make sure the context is not empty
        current_context = " " if current_context == "" else current_context

        tokens = dataset.tokenizer.encode(current_context).tokens
        tokens = tokens[-model.seq_len :]

        x = torch.tensor(model.get_token_indices(tokens)).to(device)
        x = x.unsqueeze(0)

        jpt_output, pre_output = inference_step(model, x, True)  # do final projection

        # Print the generated character

        last_token = jpt_output[0:1, -1:, :]

        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):

            # Apply temperature and sample using top-k
            logits = last_token.squeeze() / temperature

            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 50)

            # Sample from the filtered distribution
            pred_token_indices = torch.multinomial(top_k_probs, num_samples=1).unsqueeze(0)

            # Map back to original indices
            pred_token_indices = top_k_indices[pred_token_indices]

        next_token = model.get_text_token_from_indices(pred_token_indices.cpu().numpy())
        next_token = next_token.item()

        next_token = "" if next_token == "[UNK]" or next_token == "[PAD]" else next_token

        print(next_token, end="", flush=True)

        result.append(next_token)

        if len(result) > model.seq_len:
            result.pop(0)

    final_text = "".join(result)
    # print(f"\nFinal text:\n{final_text}")
    return final_text


# Update the main function to support torchrun
if __name__ == "__main__":
    # Initialize distributed environment if run with torchrun
    distributed = False
    local_rank = 0
    world_size = 1
    rank = 0
    # Check if we're running under torchrun by looking for environment variables
    if "LOCAL_RANK" in os.environ:
        distributed = True
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        # Initialize distributed process group
        setup_distributed(local_rank, world_size)

        # Set device for this process
        torch.cuda.set_device(local_rank)

        print(f"Initialized process {local_rank}/{world_size}")

    bs = 24
    # Define experiments
    experiments: list[dict] = {
        "seq_len": [1024],
        "token_space_dim": [768],
        "epochs": [2],
        "batch_size": [bs],
        "lr": [0.0008],
        "num_head": [12],
        "n_layers": [12],
        "jpt_embed_dim": [768],
        "dropout": [0.0],
        "vocab_size": [50304],
        "output_type": [JPT1ModelType.STANDARD_SAMPLED],
        "grad_accum_size": [bs * 1024 * 4 * 5],
        "log_step_size": [bs * 1024 * 4 * 5 * 2],
        "dset_ratio": [1],
        "warmup_pct": [0.03],
        "grad_accum_max_at": [0.03],
        "early_end_pct": [None],
        "total_compare_tokens": [16 * 1024],
        "beta2": [0.975],
        "weight_decay": [0.1],
    }

    experiments = create_experiments(mode="paired", **experiments)

    enable_torch_optimizations()
    setup_flash_attention()

    is_debugging = sys.gettrace() is not None

    for experiment in experiments:
        seq_len = experiment["seq_len"]
        batch_size = experiment["batch_size"]
        n_layers = experiment["n_layers"]
        num_head = experiment["num_head"]
        jpt_embed_dim = experiment["jpt_embed_dim"]
        dropout = experiment["dropout"]
        vocab_size = experiment["vocab_size"]
        output_type = experiment["output_type"]
        token_space_dim = experiment["token_space_dim"]
        grad_accum_size = experiment["grad_accum_size"]
        log_step_size = experiment["log_step_size"]
        dset_ratio = experiment["dset_ratio"]
        total_compare_tokens = experiment["total_compare_tokens"]

        dataset_name = "fineweb-10BT-edu"

        # Only load/train tokenizer on main process to avoid conflicts
        if is_main_process(distributed, local_rank):
            hf_dataset = load_hf_dataset(dataset_name)
            text_corpus_iterator = (item["text"] for item in hf_dataset["train"])
            tokenizer = get_or_train_tokenizer(
                text_corpus_iterator, vocab_size, f"tokenizer_cache/{dataset_name}_tokenizer_{vocab_size}.json"
            )

        if distributed:
            torch.distributed.barrier()

        hf_dataset = load_hf_dataset(dataset_name)
        # Other processes wait for main process to finish tokenizer
        tokenizer = Tokenizer.from_file(f"tokenizer_cache/{dataset_name}_tokenizer_{vocab_size}.json")

        loss_fn = None

        if output_type == JPT1QuantModelType.STANDARD_SAMPLED:
            loss_fn = CustomSampledLoss(ignore_index=tokenizer.token_to_id("[PAD]"), total_compare_tokens=total_compare_tokens)
        else:
            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

        dataset_train = Fineweb10BDataset(
            seq_len=seq_len,
            type="train",
            data_stride=seq_len,
            tokenizer=tokenizer,
            hf_dataset=hf_dataset,
            dataset_name=dataset_name,
            dset_ratio=dset_ratio,
        )

        vocab_size = len(tokenizer.get_vocab())

        val_dataset = Fineweb10BDataset(
            seq_len=seq_len,
            type="validation",
            data_stride=seq_len,
            tokenizer=tokenizer,
            hf_dataset=hf_dataset,
            dataset_name=dataset_name,
        )

        # Create model
        gpt_model = JPT1Quantized(
            token_space_dim=token_space_dim,
            seq_len=seq_len,
            embed_dim=jpt_embed_dim,
            num_head=num_head,
            num_layers=n_layers,
            dropout=dropout,
            tokenizer=tokenizer,
            model_type=output_type,
        )

        # Only compile model if not debugging and not distributed
        # (compiled models can cause issues with distributed training)
        should_compile = sys.gettrace() is None
        if should_compile:
            print("Compiling models...")
            gpt_model = torch.compile(gpt_model)
            loss_fn = torch.compile(loss_fn)
            print("Models compiled!")

        project_name = "jpt1"
        exp_name = f"{project_name}-sl:{experiment['seq_len']}-e:{experiment['epochs']}-bs:{experiment['batch_size']}-lr:{experiment['lr']}-hs:{experiment['num_head']}-nl:{experiment['n_layers']}-ed:{experiment['jpt_embed_dim']}-ts:{experiment['token_space_dim']}"

        # Create distributed samplers if running distributed

        train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=local_rank, shuffle=True)

        # val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)

        train_dataloader = DataLoader(
            dataset_train,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,  # Reduced for distributed setting
            pin_memory=True,
            prefetch_factor=12,  # Reduced for distributed setting
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            # sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=12,
        )

        # Create wrapper function for train_model
        def train_model_lambda(wandb):
            model = train_model(
                wandb,
                gpt_model,
                train_dataloader,
                val_dataloader,
                experiment,
                loss_fn,
                distributed,
                local_rank,
            )
            return model

        run_experiment(project_name, train_model_lambda, exp_name, distributed, local_rank, experiment)

    # Clean up distributed resources
    if distributed:
        cleanup_distributed()

print("Training Complete")
