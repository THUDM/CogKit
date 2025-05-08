import torch
from pydantic import BaseModel


class BaseState(BaseModel):
    # Allow arbitrary types (for torch dtype)
    model_config = {"arbitrary_types_allowed": True}

    world_size: int
    local_rank: int
    global_rank: int

    device: torch.device

    weight_dtype: torch.dtype

    train_steps: int = -1
    train_epochs: int = -1
    num_trainable_parameters: int = -1
    num_update_steps_per_epoch: int = -1
    total_batch_size_count: int = -1

    generator: torch.Generator | None = None
