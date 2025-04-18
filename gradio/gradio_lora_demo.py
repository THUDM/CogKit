import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch
from datasets import Dataset
from slugify import slugify
from torchvision.io import write_video
from utils import (
    BaseTask,
    get_dataset_dirs,
    get_logger,
    get_lora_checkpoint_rootdir,
    get_resolutions,
    get_training_script,
    load_config_template,
    load_data,
    resolve_path,
)

import gradio as gr
from cogkit import GenerationMode, guess_generation_mode
from cogkit.utils import flatten_dict

# =======================  global state  ====================

logger = get_logger(__name__)

data_dirs: List[str] = get_dataset_dirs()
checkpoint_rootdir: str = ""
checkpoint_name: str = ""
checkpoint_dir: str | None = None

task: GenerationMode | None = None
task_config: Dict[str, Any] = {}

train_data: Dataset | None = None
test_data: Dataset | None = None

resolution: str | None = None

current_training_task = None

# =========================  hooks  =========================


def update_lora_name(name: str) -> Tuple[gr.Textbox]:
    """Update the lora_name when the text field changes."""
    global checkpoint_dir, checkpoint_name, lora_name
    lora_name.value = name
    checkpoint_name = slugify(name)

    checkpoint_dir = checkpoint_rootdir + "/" + checkpoint_name
    updated_checkpoint_dir = gr.Textbox(
        label="Checkpoint Directory",
        info="Path to the checkpoint directory",
        interactive=False,
        value=checkpoint_dir,
    )
    return updated_checkpoint_dir


def update_task(hf_model_id: str) -> Tuple[gr.Dropdown]:
    global task, task_config, checkpoint_rootdir, checkpoint_dir, model_type, resolution
    model_type.value = hf_model_id
    task = guess_generation_mode("THUDM/" + hf_model_id)
    task_config = load_config_template(task)

    checkpoint_rootdir = get_lora_checkpoint_rootdir(task)
    checkpoint_dir = checkpoint_rootdir + "/" + checkpoint_name

    if task == GenerationMode.TextToImage:
        resolution_list = get_resolutions(task)
        default_value = resolution_list[0]
        info = "Height x Width"

    elif task == GenerationMode.TextToVideo:
        resolution_list = get_resolutions(task)
        default_value = resolution_list[0]
        info = "Frames x Height x Width"

    logger.info(f"Current task: {task}")
    logger.info(f"lora_checkpoint_rootdir: {checkpoint_rootdir}")

    updated_checkpoint_dir = gr.Textbox(
        label="Checkpoint Directory",
        info="Path to the checkpoint directory",
        interactive=False,
        value=checkpoint_dir,
    )

    resolution = default_value
    updated_resolution_dropdown = gr.Dropdown(
        choices=resolution_list,
        label="Training Resolution",
        info=info,
        interactive=True,
        value=default_value,
    )

    # Return the resolution list for updating train_resolution choices
    return updated_checkpoint_dir, updated_resolution_dropdown


def update_do_validation(user_input_do_validation: bool) -> None:
    global do_validation
    do_validation.value = user_input_do_validation


def update_train_data(user_input_data_dir: str) -> List[Tuple[str, str]]:
    global train_data, data_dir
    assert task is not None

    data_dir.value = user_input_data_dir

    progress = gr.Progress()
    progress(0, desc="Loading dataset...")
    train_data = load_data(user_input_data_dir, task)
    progress(1, desc="Dataset loaded successfully!")
    logger.info(f"Loaded training data from {user_input_data_dir}")
    logger.info(f"Train data: {train_data}")

    ###### Prepare data for display in the gallery component
    if task == GenerationMode.TextToImage:
        # num_samples = min(10, len(train_data))
        num_samples = len(train_data)
        sample_images = train_data["image"][:num_samples]
        sample_captions = train_data["prompt"][:num_samples]
        return [(img, cap) for img, cap in zip(sample_images, sample_captions)]

    elif task == GenerationMode.TextToVideo:
        # Create a temporary directory to store video files
        temp_dir = Path(tempfile.gettempdir()) / "cogkit_videos"
        os.makedirs(temp_dir, exist_ok=True)

        num_samples = min(50, len(train_data))
        sample_videos = []
        sample_captions = train_data["prompt"][:num_samples]

        # Save videos to temporary files
        for i, video in enumerate(train_data["video"][:num_samples]):
            video_path = str(temp_dir / f"{uuid.uuid4()}.mp4")

            # Get frames from VideoReader and convert to tensor
            frames = []
            for frame in video:
                frames.append(frame["data"])

            # Stack frames and save as video
            if frames:
                video_tensor = torch.stack(frames)
                # Change from (T, C, H, W) to (T, H, W, C) format
                video_tensor = video_tensor.permute(0, 2, 3, 1)
                fps = video.get_metadata().get("fps", 30)  # Default to 30 fps if not available
                write_video(video_path, video_tensor, fps=fps)
                sample_videos.append(video_path)

        return [(video_path, cap) for video_path, cap in zip(sample_videos, sample_captions)]

    return []


def update_training_config() -> None:
    assert model_type.value is not None
    assert task_config["model"]["model_type"] == task.value
    assert task_config["model"]["training_type"] == "lora"

    if lora_name.value is None or lora_name.value == "":
        raise gr.Error("Lora name cannot be empty")

    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ###### Rewrite configs
    task_config["model"]["model_path"] = "THUDM/" + model_type.value

    task_config["output"]["output_dir"] = resolve_path(out_dir)

    task_config["data"]["data_root"] = data_dir.value

    task_config["training"]["train_epochs"] = epochs.value
    task_config["training"]["batch_size"] = batch_size.value
    task_config["training"]["learning_rate"] = learning_rate.value
    task_config["training"]["train_resolution"] = resolution

    task_config["checkpoint"]["checkpointing_steps"] = checkpointing_step.value
    task_config["checkpoint"]["checkpointing_limit"] = checkpointing_limit.value

    task_config["validation"]["do_validation"] = do_validation.value
    task_config["validation"]["validation_steps"] = checkpointing_step.value

    logger.info(f"task config: {task_config}")


def run_training() -> BaseTask:
    """Run the training process using accelerate launch and the configured parameters."""
    logger.info("Starting training process...")

    # Verify task is initialized
    if task is None:
        raise gr.Error("Error: No model type selected")

    # Verify data directory is set
    if not train_data:
        raise gr.Error("Error: No training data loaded")

    cmd_args = [sys.executable, get_training_script()]

    # Flatten command line dict
    flat_config = flatten_dict(task_config, ignore_none=True)

    for param_name, param_value in flat_config.items():
        cmd_args.extend([f"--{param_name}", str(param_value)])

    # Create and run the task
    training_task = BaseTask(cmd_args)
    training_task.run()

    return training_task


def start_training_process() -> Iterator[str]:
    """Update the training config and start the training process."""
    global current_training_task

    # First update the training configuration
    update_training_config()

    # Then run the training
    current_training_task = run_training()

    gr.Info(f"Training process started with PID: {current_training_task.get_pid()}")

    # Stream output from the task
    output_text = ""
    for line in current_training_task.iter_output():
        output_text += line + "\n"
        yield output_text

    # Final update after process completes
    yield output_text + "\nTraining process completed!"

    gr.Info("Training process completed!")


def update_training_resolution(user_input_resolution: str) -> None:
    """Update the training resolution in the task config."""
    global resolution
    resolution = user_input_resolution
    logger.info(f"Updating training resolution: {resolution}")


# ===========================  UI  ===========================

with gr.Blocks() as demo:
    # gr.Markdown("""# LoRA Ease for CogView 🧞‍♂️""")

    with gr.Row():
        lora_name = gr.Textbox(
            label="Name of your LoRA checkpoint",
            info="This has to be a unique name",
            placeholder="e.g.: Persian Miniature Painting style, Cat Toy",
            value="",
        )

        model_type = gr.Dropdown(
            choices=["CogView4-6B", "CogVideoX1.5-5B"],
            label="Model",
            info="Select the model to use",
            interactive=True,
            value=None,
        )

        data_dir = gr.Dropdown(
            choices=data_dirs,
            label="Dataset Directory",
            info="Select the dataset directory to use",
            interactive=True,
            value=None,
        )

        checkpoint_dir = gr.Textbox(
            label="Checkpoint Directory",
            info="Path to the checkpoint directory",
            interactive=False,
            value=None,
        )

    # Add a section to display training data samples
    with gr.Row():
        sample_gallery = gr.Gallery(
            label="Training Data Preview",
            show_label=True,
            elem_id="gallery",
            columns=5,
            object_fit="contain",
            height="auto",
        )

    gr.Markdown("### Training Configuration")
    with gr.Column():
        with gr.Row():
            train_resolution = gr.Dropdown(
                choices=[],
                label="Training Resolution",
                info="Resolution for training",
                interactive=True,
            )
            batch_size = gr.Number(
                value=1,
                label="Batch Size",
                info="Number of samples per training batch",
                interactive=True,
            )
            epochs = gr.Number(
                value=1,
                label="Epochs",
                info="Number of training epochs",
                interactive=True,
            )
            learning_rate = gr.Number(
                value=2e-5,
                step=1e-5,
                label="Learning Rate",
                info="Training learning rate",
                interactive=True,
            )
            checkpointing_step = gr.Number(
                value=10,
                label="Checkpointing Step",
                info="Number of training steps between checkpoints",
                interactive=True,
            )
            checkpointing_limit = gr.Number(
                value=2,
                label="Checkpointing Limit",
                info="Maximum number of checkpoints to keep",
                interactive=True,
            )

        do_validation = gr.Checkbox(
            value=False,
            label="Enable Validation",
            info="Whether to perform validation during training",
        )

    start_train = gr.Button("Start training")

    training_output = gr.Textbox(
        label="Training Output",
        placeholder="Training output will appear here...",
        interactive=False,
        lines=20,
        autoscroll=True,
    )

    ###### Binding hooks
    lora_name.change(fn=update_lora_name, inputs=[lora_name], outputs=[checkpoint_dir])
    model_type.change(
        fn=update_task, inputs=[model_type], outputs=[checkpoint_dir, train_resolution]
    )
    train_resolution.change(fn=update_training_resolution, inputs=[train_resolution])
    data_dir.change(fn=update_train_data, inputs=[data_dir], outputs=[sample_gallery])
    do_validation.change(fn=update_do_validation, inputs=[do_validation])
    start_train.click(fn=start_training_process, inputs=None, outputs=[training_output])


if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
