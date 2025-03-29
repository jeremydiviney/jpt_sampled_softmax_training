import time
import glob
import os
import psutil

from torch import nn
import torch

import wandb
from helpers.distributed_utils import is_main_process


def get_gpu_memory_gb() -> float:
    """Get current GPU memory usage in GB for the current device"""
    if not torch.cuda.is_available():
        return 0.0

    return float(torch.cuda.max_memory_allocated()) / 1e9  # Convert bytes to GB


def get_memory_gb() -> float:
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024 * 1024)


def count_parameters(model: nn.Module) -> tuple[int, dict]:
    """
    Count total trainable parameters and parameters per layer
    Returns: (total_params, params_by_layer)
    """
    params_by_layer = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    total_params = sum(params_by_layer.values())

    # Format large numbers with commas
    formatted_layers = {name: f"{count:,}" for name, count in params_by_layer.items()}

    print(f"\nTotal trainable parameters: {total_params:,}")
    # print("\nParameters per layer:")
    # for name, count in formatted_layers.items():
    #     print(f"{name}: {count}")

    return total_params, params_by_layer


def save_project_files_as_artifact(wandb_run):
    """Save all Python files in project as artifacts"""
    # Create an artifact
    artifact = wandb.Artifact(
        name=f"source_code_{wandb_run.id}",
        type="code",
        description="Source code for this run",
    )

    # Include all Python files
    include_pattern = "**/*.py"
    # Exclude patterns for specific directories
    exclude_patterns = [
        ".venv/**/*.py",
        ".vscode/**/*.py",
        "wandb/**/*.py",
        "**/__init__.py",
    ]

    # Get all Python files then filter out excluded paths
    python_files = set(glob.glob(include_pattern, recursive=True)) - set(
        file for pattern in exclude_patterns for file in glob.glob(pattern, recursive=True)
    )

    # Add files to artifact
    for file_path in python_files:
        artifact.add_file(file_path)

    # Log the artifact
    wandb_run.log_artifact(artifact)


def run_experiment(projectName, train_model, exp_name, distributed, local_rank, config: dict) -> None:

    if is_main_process(distributed, local_rank):

        wandb_key = os.environ.get("WANDB_API_KEY")

        if wandb_key is None:
            raise ValueError("WANDB_API_KEY is not set, please set it in the .env file")

        # Initialize wandb
        wandb.login(key=wandb_key)

        wandb.init(
            project=projectName,
            config=config,
            name=exp_name,
        )

        # Track time and memory
        start_time = time.time()

        # Save source code at start of run
        save_project_files_as_artifact(wandb.run)

    # Train model and get parameters count
    train_model = train_model(wandb)

    total_params, params_by_layer = count_parameters(train_model)

    if is_main_process(distributed, local_rank):

        # Log parameters count
        wandb.log({"total_parameters": total_params, "run_id": wandb.run.id})

        gpu_memory_usage = get_gpu_memory_gb()

        # Log time and memory metrics
        end_time = time.time()
        duration = end_time - start_time

        wandb.log(
            {
                "duration_seconds": duration,
                "duration_per_epoch_seconds": duration / config["epochs"],
                "gpu_memory_usage_gb": gpu_memory_usage,
            }
        )

        wandb.finish()


def create_experiments(mode="cartesian", **param_lists):
    """
    Create a list of experiment configurations.

    Args:
        mode: How to combine parameter values:
              - "cartesian": Generate all possible combinations (Cartesian product)
              - "paired": Use corresponding elements from each parameter list
        **param_lists: Dictionary of parameter names to lists of values

    Returns:
        List of experiment configurations
    """
    if mode == "paired":
        # Check that all multi-value parameters have the same length
        multi_value_params = [(name, values) for name, values in param_lists.items() if len(values) > 1]
        if multi_value_params:
            expected_length = len(multi_value_params[0][1])
            mismatched_params = [name for name, values in multi_value_params if len(values) != expected_length]

            if mismatched_params:
                raise ValueError(
                    f"All multi-value parameters must have the same length when mode='paired'. "
                    f"Mismatched parameters: {mismatched_params}"
                )

        # Find the length to use for pairing (max length of any parameter list)
        pair_length = max([len(values) for values in param_lists.values()], default=1)

        # Create experiments with paired parameters
        experiments = []
        for i in range(pair_length):
            experiment = {}
            for param_name, param_values in param_lists.items():
                # If parameter has only one value, use it for all experiments
                # Otherwise, use the value at the current index
                experiment[param_name] = param_values[i % len(param_values)]
            experiments.append(experiment)
        return experiments

    elif mode == "cartesian":
        # Generate all combinations (Cartesian product)
        # Start with a single empty experiment
        experiments = [{}]

        # For each parameter, create new experiments with all possible values
        for param_name, param_values in param_lists.items():
            new_experiments = []
            for experiment in experiments:
                for value in param_values:
                    new_experiment = experiment.copy()
                    new_experiment[param_name] = value
                    new_experiments.append(new_experiment)
            experiments = new_experiments

        return experiments

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'cartesian' or 'paired'.")
