import os
from typing import Any

import torch
import torch.distributed as dist


def check_distributed() -> None:
    if not dist.is_initialized():
        raise RuntimeError("Distributed training is not initialized")


def is_main_process() -> bool:
    return dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size()


def get_global_rank() -> int:
    return dist.get_rank()


def get_local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def get_device() -> torch.device:
    return torch.device(f"cuda:{get_local_rank()}")


def gather_object(object: Any) -> list[Any]:
    output_objects = [None for _ in range(get_world_size())]
    dist.all_gather_object(output_objects, object)
    return output_objects
