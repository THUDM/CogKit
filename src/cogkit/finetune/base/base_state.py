import torch
from pydantic import BaseModel


class BaseState(BaseModel):
    # Allow arbitrary types (for torch dtype)
    model_config = {"arbitrary_types_allowed": True}

    weight_dtype: torch.dtype = torch.float32  # dtype for mixed precision training
    num_trainable_parameters: int = 0
    num_update_steps_per_epoch: int = 0
    total_batch_size_count: int = 0

    generator: torch.Generator | None = None

    using_deepspeed: bool = False
