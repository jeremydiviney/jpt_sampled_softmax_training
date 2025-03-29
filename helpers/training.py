from typing import TypeVar, Any, Optional
import os
import torch
import torch.nn as nn

Model = TypeVar("Model", bound=nn.Module)


def check_memory_usage(model: nn.Module):
    if torch.cuda.is_available():
        current_mem = float(torch.cuda.memory_allocated()) / 1e9
        # max_mem = float(torch.cuda.max_memory_allocated())/1e9
        # print("Current memory: {:.2f}GB".format(current_mem))
        # print("Max memory: {:.2f}GB".format(max_mem))
        model.average_memory_usage = (model.average_memory_usage + current_mem) / 2
        # print("Average memory usage: {:.2f}GB".format(self.average_memory_usage))


def save_model(model: Any, save_dir: str, model_name: str) -> None:
    """
    Save the model state. Optionally save encoder and decoder separately.

    Args:
        model: The model to save
        save_dir: Directory to save the model(s) in
        model_name: Base name for the saved model files
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save full model
    full_model_path = os.path.join(save_dir, f"{model_name}.pt")
    torch.save(model.state_dict(), full_model_path)


def load_model(model: Model, load_dir: str, model_name: str, device: Optional[str] = None) -> Model:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    def clean_state_dict(state_dict: dict) -> dict:
        return {key.replace("_orig_mod.", ""): value for key, value in state_dict.items()}

    full_model_path = os.path.join(load_dir, f"{model_name}_full.pt")
    state_dict = torch.load(full_model_path, map_location=device)
    model.load_state_dict(clean_state_dict(state_dict))
    return model.to(device)


# 2. Enable torch.backends optimizations
def enable_torch_optimizations():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        # Enable TF32 for faster matrix multiplications
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cudnn benchmarking
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True


def setup_flash_attention():
    # Enable Flash Attention if available
    if torch.cuda.is_available():
        # flash_available = torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)
        # flash_available = torch.nn.attention.sdpa_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True)

        print(f"Flash Attention available and enabled")
        # Enable Flash Attention
        torch.backends.cuda.enable_flash_sdp(True)
        # Enable Math Flash Attention (more efficient math operations)
        torch.backends.cuda.enable_math_sdp(True)
        # Enable Memory Efficient Attention
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        # return flash_available
        return True
    print("CUDA not available, Flash Attention disabled")
    return False
