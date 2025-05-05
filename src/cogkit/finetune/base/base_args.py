# -*- coding: utf-8 -*-

import datetime
import logging
from datetime import timedelta
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationInfo, field_validator


class BaseArgs(BaseModel):
    model_config = {"frozen": True, "extra": "ignore"}

    ########## Logging ##########
    name4train: str
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    ########## Model ##########
    model_path: Path
    model_name: str

    ########## Output ##########
    output_dir: Path = Path(f"train_result/{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}")

    ########## Tracker ##########
    report_to: Literal["wandb"] | None = None

    ########## Data Path ###########
    data_root: Path

    ########## Training #########
    training_type: Literal["lora", "sft"] = "lora"
    strategy: Literal[
        "DDP", "SHARD_GRAD_OP", "FULL_SHARD", "HYBRID_SHARD", "_HYBRID_SHARD_ZERO2"
    ] = "FULL_SHARD"
    # This will offload model param and grads to CPU memory to save GPU memory, but will slow down training
    offload_params_grads: bool = False
    # This will increase memory usage since gradients are sharded during accumulation step.
    # Note, when used with offload_params_grads, model parameters and gradients will only be offloaded
    #   to the CPU during the final synchronization (still retained on GPU in gradient accumulation steps)
    #   which means offload_params_grads is meaningless when used with no_grad_sync_when_accumulating
    no_grad_sync_when_accumulating: bool = False

    resume_from_checkpoint: Path | None = None

    seed: int | None = None
    train_epochs: int
    checkpointing_steps: int
    checkpointing_limit: int

    batch_size: int
    gradient_accumulation_steps: int = 1

    mixed_precision: Literal["fp32", "fp16", "bf16"]
    low_vram: bool = False

    learning_rate: float = 2e-5
    optimizer: str = "adamw"
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0

    lr_scheduler: str = "CosineAnnealingLR"

    num_workers: int = 8
    pin_memory: bool = True

    gradient_checkpointing: bool = True
    nccl_timeout: timedelta = timedelta(seconds=1800)

    ########## Validation ##########
    do_validation: bool = False
    validation_steps: int | None  # if set, should be a multiple of checkpointing_steps

    @field_validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        match v:
            case "DEBUG":
                return logging.DEBUG
            case "INFO":
                return logging.INFO
            case "WARNING":
                return logging.WARNING
            case "ERROR":
                return logging.ERROR
            case "CRITICAL":
                return logging.CRITICAL
            case _:
                raise ValueError("log_level must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL")

    @field_validator("nccl_timeout")
    def validate_nccl_timeout(cls, v: timedelta | int) -> timedelta:
        if isinstance(v, int):
            return timedelta(seconds=v)
        return v

    @field_validator("low_vram")
    def validate_low_vram(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data.get("training_type") != "lora":
            raise ValueError("low_vram can only be True when training_type is 'lora'")
        if v and info.data.get("offload_params_grads"):
            raise ValueError("low_vram and offload_params_grads cannot be enabled simultaneously")
        if v and info.data.get("strategy") != "DDP":
            raise ValueError("low_vram can only be used with strategy='DDP'")
        if v and info.data.get("resume_from_checkpoint") is not None:
            raise ValueError("resume_from_checkpoint cannot be used when low_vram is True")
        return v

    @field_validator("strategy")
    def validate_strategy(cls, v: str, info: ValidationInfo) -> str:
        if info.data.get("training_type") == "lora" and v != "DDP":
            raise ValueError("When using lora training_type, strategy must be 'DDP'")
        return v

    @field_validator("offload_params_grads")
    def validate_offload_params_grads(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data.get("low_vram"):
            raise ValueError("low_vram and offload_params_grads cannot be enabled simultaneously")
        if v and info.data.get("no_grad_sync_when_accumulating"):
            raise ValueError(
                "offload_params_grads and no_grad_sync_when_accumulating cannot be enabled simultaneously"
            )
        if v and info.data.get("strategy") == "DDP":
            raise ValueError("offload_params_grads cannot be enabled when strategy is 'DDP'")
        return v

    @field_validator("no_grad_sync_when_accumulating")
    def validate_no_grad_sync_when_accumulating(cls, v: bool, info: ValidationInfo) -> bool:
        if v and info.data.get("offload_params_grads"):
            raise ValueError(
                "offload_params_grads and no_grad_sync_when_accumulating cannot be enabled simultaneously"
            )
        if v and info.data.get("strategy") == "DDP":
            raise ValueError(
                "no_grad_sync_when_accumulating cannot be enabled when strategy is 'DDP'"
            )
        return v

    @field_validator("validation_steps")
    def validate_validation_steps(cls, v: int | None, info: ValidationInfo) -> int | None:
        values = info.data
        if values.get("do_validation"):
            if v is None:
                raise ValueError("validation_steps must be specified when do_validation is True")
            if values.get("checkpointing_steps") and v % values["checkpointing_steps"] != 0:
                raise ValueError("validation_steps must be a multiple of checkpointing_steps")
        return v

    @field_validator("mixed_precision")
    def validate_mixed_precision(cls, v: str, info: ValidationInfo) -> str:
        if v == "fp16" and "cogvideox-2b" not in str(info.data.get("model_path", "")).lower():
            logging.warning(
                "All CogVideoX models except cogvideox-2b were trained with bfloat16. "
                "Using fp16 precision may lead to training instability."
            )
        return v

    @classmethod
    def parse_from_yaml(cls, fpath: str | Path) -> "BaseArgs":
        if isinstance(fpath, str):
            fpath = Path(fpath)

        with open(fpath, "r") as f:
            yaml_dict = yaml.safe_load(f)

        return cls(**yaml_dict)
