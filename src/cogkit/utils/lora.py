# -*- coding: utf-8 -*-

"""
LoRA utility functions for model fine-tuning.

This module provides a user-friendly interface for working with LoRA adapters
based on Huggings PEFT (Parameter-Efficient Fine-Tuning) library.
It simplifies the process of injecting, saving, loading, and unloading LoRA
adapters for transformer models.

For more details, refer to: https://huggingface.co/docs/peft/developer_guides/low_level_api
"""

from pathlib import Path

from peft import (
    LoraConfig,
    get_peft_model_state_dict,
    inject_adapter_in_model,
    set_peft_model_state_dict,
)
from safetensors.torch import load_file, save_file

from diffusers.loaders import CogVideoXLoraLoaderMixin, CogView4LoraLoaderMixin
from diffusers.utils import recurse_remove_peft_layers

# Standard filename for LoRA adapter weights
_LORA_WEIGHT_NAME = "adapter_model.safetensors"


def _get_lora_config() -> LoraConfig:
    return LoraConfig(
        r=128,
        lora_alpha=64,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )


def inject_lora(model, lora_dir_or_state_dict: str | Path | None = None) -> None:
    """
    Inject LoRA adapters into the model.

    This function adds LoRA layers to the specified model. If a LoRA checkpoint
    is provided, it will load the weights from that checkpoint. Otherwise, it
    will initialize the LoRA weights randomly.

    Args:
        model: The model to inject LoRA adapters into
        lora_dir_or_state_dict: Path to a LoRA checkpoint directory, a state dict,
                                or None for random initialization
    """
    transformer_lora_config = _get_lora_config()
    inject_adapter_in_model(transformer_lora_config, model)
    if lora_dir_or_state_dict is None:
        return

    if isinstance(lora_dir_or_state_dict, str) or isinstance(lora_dir_or_state_dict, Path):
        lora_dir = Path(lora_dir_or_state_dict)
        lora_fpath = lora_dir / _LORA_WEIGHT_NAME
        assert lora_dir.exists(), f"LORA checkpoint directory {lora_dir} does not exist"
        assert lora_fpath.exists(), f"LORA checkpoint file {lora_fpath} does not exist"

        peft_state_dict = load_file(lora_fpath, device="cpu")
    else:
        peft_state_dict = lora_dir_or_state_dict

    set_peft_model_state_dict(model, peft_state_dict)


def save_lora(model, lora_dir: str | Path) -> None:
    """
    Save the LoRA adapter weights from a model to disk.

    Args:
        model: The model containing LoRA adapters to save
        lora_dir: Directory path where the LoRA weights will be saved

    Raises:
        ValueError: If no LoRA weights are found in the model
    """
    lora_dir = Path(lora_dir)
    peft_state_dict = get_peft_model_state_dict(model)
    if not peft_state_dict:
        raise ValueError("No LoRA weights found in the model")

    lora_fpath = lora_dir / _LORA_WEIGHT_NAME
    save_file(peft_state_dict, lora_fpath, metadata={"format": "pt"})


def unload_lora(model) -> None:
    """
    Remove all LoRA adapters from the model.

    This function recursively removes all PEFT (LoRA) layers from the model,
    returning it to its original state without the adapters.

    Args:
        model: The model from which to remove LoRA adapters
    """
    recurse_remove_peft_layers(model)


def load_lora_checkpoint(
    pipeline: CogVideoXLoraLoaderMixin | CogView4LoraLoaderMixin,
    lora_dir: str | Path,
) -> None:
    """
    Load a LoRA checkpoint into a pipeline.

    This is a convenience function that injects LoRA adapters into the transformer
    component of the specified pipeline and loads the weights from the checkpoint.
    """
    lora_dir = Path(lora_dir)
    inject_lora(pipeline.transformer, lora_dir)


def unload_lora_checkpoint(
    pipeline: CogVideoXLoraLoaderMixin | CogView4LoraLoaderMixin,
) -> None:
    """
    Remove LoRA adapters from a pipeline.

    This is a convenience function that removes all LoRA adapters from the
    transformer component of the specified pipeline.
    """
    unload_lora(pipeline.transformer)
