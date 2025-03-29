import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed(rank, world_size):
    """
    Setup distributed training environment
    """
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"Initialized process group with rank {rank} and world size {world_size}")


def cleanup_distributed():
    """
    Clean up distributed training environment
    """
    dist.destroy_process_group()


def get_model_for_training(model, device, distributed=False, local_rank=None):
    """
    Prepare model for training - move to device and wrap in DDP if needed

    Args:
        model: The model to prepare
        device: The device to move the model to
        distributed: Whether to use distributed training
        local_rank: The local GPU rank in distributed mode

    Returns:
        The prepared model
    """
    model = model.to(device)

    if distributed:
        # Convert BatchNorm to SyncBatchNorm if necessary
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # Wrap model with DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    return model


def is_main_process(distributed=False, rank=0):
    """
    Check if this is the main process (for saving checkpoints, logging, etc.)
    """
    if distributed:
        return rank == 0
    return True


def get_world_size(distributed=False):
    """
    Get the number of processes
    """
    if distributed and dist.is_initialized():
        return dist.get_world_size()
    return 1


def reduce_value(value, average=True):
    """
    Reduce a value across all processes
    """
    if not torch.is_tensor(value):
        value = torch.tensor(value)

    if not dist.is_initialized():
        return value

    world_size = dist.get_world_size()
    if world_size < 2:
        return value

    with torch.no_grad():
        dist.all_reduce(value)
        if average:
            value = value / world_size

    return value
