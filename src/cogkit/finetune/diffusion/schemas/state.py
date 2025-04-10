# -*- coding: utf-8 -*-


from pathlib import Path
from typing import Any

import torch

from cogkit.finetune.base import BaseState


class DiffusionState(BaseState):
    transformer_config: dict[str, Any] = None

    # for video input, train_resolution = (frames, height, width)
    # for image input, train_resolution = (height, width)
    train_resolution: tuple[int, int, int] | tuple[int, int]

    # packing realted
    max_vtoken_length: int | None = None
    training_seq_length: int | None = None

    validation_prompts: list[str] = []
    validation_images: list[Path | None] = []
    validation_videos: list[Path | None] = []

    negative_prompt_embeds: torch.Tensor | None = None
