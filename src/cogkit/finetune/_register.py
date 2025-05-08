# -*- coding: utf-8 -*-


from typing import TYPE_CHECKING, Literal

# using TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from cogkit.finetune import BaseTrainer

SUPPORTED_MODELS: dict[str, dict[str, "BaseTrainer"]] = {}


def register(
    model_name: str,
    training_type: Literal["lora", "sft"],
    trainer_cls: "BaseTrainer",
):
    """Register a model and its associated functions for a specific training type.

    Args:
        model_name (str): Name of the model to register (e.g. "cogvideox-5b")
        training_type (Literal["lora", "sft"]): Type of training - either "lora" or "sft"
        trainer_cls (Trainer): Trainer class to register.
    """

    # Check if model_name and training_type exists in SUPPORTED_MODELS
    if model_name not in SUPPORTED_MODELS:
        SUPPORTED_MODELS[model_name] = {}
    elif training_type in SUPPORTED_MODELS[model_name]:
        raise ValueError(f"Training type {training_type} already exists for model {model_name}")

    SUPPORTED_MODELS[model_name][training_type] = trainer_cls


def show_supported_models():
    """Print all currently supported models and their training types."""

    print("\nSupported Models:")
    print("================")

    for model_name, training_types in SUPPORTED_MODELS.items():
        print(f"\n{model_name}")
        print("-" * len(model_name))
        for training_type in training_types:
            print(f"  • {training_type}")


def get_model_cls(
    model_name: str, training_type: Literal["lora", "sft"], use_packing: bool = False
) -> "BaseTrainer":
    """Get the trainer class for a specific model and training type."""
    if model_name not in SUPPORTED_MODELS:
        print(f"\nModel '{model_name}' is not supported.")
        print("\nSupported models are:")
        for supported_model in SUPPORTED_MODELS:
            print(f"  • {supported_model}")
        raise ValueError(f"Model '{model_name}' is not supported")

    if use_packing:
        training_type = f"{training_type}-packing"

    if training_type not in SUPPORTED_MODELS[model_name]:
        print(f"\nTraining type '{training_type}' is not supported for model '{model_name}'.")
        print(f"\nSupported training types for '{model_name}' are:")
        for supported_type in SUPPORTED_MODELS[model_name]:
            print(f"  • {supported_type}")
        raise ValueError(
            f"Training type '{training_type}' is not supported for model '{model_name}'"
        )

    return SUPPORTED_MODELS[model_name][training_type]
