from typing import Any

import torch.distributed as dist
import wandb

from .dist import is_main_process


class WandbTracker:
    def __init__(self, name: str, config: dict[str, Any], **kwargs: Any) -> None:
        if is_main_process():
            self.tracker = wandb.init(
                name=name,
                config=config,
                **kwargs,
            )
        dist.barrier()

    def log(self, *args: Any, **kwargs: Any) -> None:
        if is_main_process():
            self.tracker.log(*args, **kwargs)
        dist.barrier()

    def finish(self) -> None:
        if is_main_process():
            self.tracker.finish()
        dist.barrier()
