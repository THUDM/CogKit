import os
import tempfile
import uuid
from pathlib import Path
from typing import List, Tuple

import torch
from utils import (
    get_logger,
    get_lora_checkpoint_dirs,
    get_lora_checkpoint_rootdir,
    get_resolutions,
)

import gradio as gr
from cogkit import (
    GenerationMode,
    generate_image,
    generate_video,
    guess_generation_mode,
    load_lora_checkpoint,
    load_pipeline,
    unload_lora_checkpoint,
)
from diffusers.utils import export_to_video

# =======================  global state  ====================

logger = get_logger(__name__)

task: GenerationMode | None = None
checkpoint_rootdir: str = ""
checkpoint_dirs: List[str] = []
pipeline = None

prev_model_id: str | None = None
resolution: str | None = None

# =========================  hooks  =========================


def update_task(hf_model_id: str) -> Tuple[gr.Dropdown, gr.Component]:
    """Update the task based on model selection and load available LoRA checkpoints."""
    global task, checkpoint_rootdir, checkpoint_dirs, resolution

    task = guess_generation_mode("THUDM/" + hf_model_id)
    checkpoint_rootdir = get_lora_checkpoint_rootdir(task)

    # Get all available checkpoints for the selected task
    checkpoint_dirs = get_lora_checkpoint_dirs(task)

    # Add a "None" option at the beginning for no LoRA
    checkpoint_options = ["None"] + checkpoint_dirs

    logger.info(f"Current task: {task}")
    logger.info(f"Checkpoint root dir: {checkpoint_rootdir}")
    logger.info(f"Available checkpoints: {checkpoint_options}")

    updated_lora_dropdown = gr.Dropdown(
        choices=checkpoint_options,
        label="LoRA Checkpoint",
        info="Select a LoRA checkpoint to use for inference",
        interactive=True,
        value="None",
    )

    # Reset the subcheckpoint dropdown
    updated_subcheckpoint_dropdown = gr.Dropdown(
        choices=[],
        label="Checkpoint Version",
        info="Select a specific checkpoint version",
        interactive=False,
        value=None,
        visible=False,
    )

    # Configure resolution dropdown based on task
    if task == GenerationMode.TextToImage:
        resolution_list = get_resolutions(task)
        default_resolution = resolution_list[0]
        resolution_info = "Height x Width"
    elif task == GenerationMode.TextToVideo:
        resolution_list = get_resolutions(task)
        default_resolution = resolution_list[0]
        resolution_info = "Frames x Height x Width"
    else:
        resolution_list = []
        default_resolution = None
        resolution_info = ""

    resolution = default_resolution

    updated_resolution_dropdown = gr.Dropdown(
        choices=resolution_list,
        label="Resolution",
        info=resolution_info,
        interactive=True,
        value=default_resolution,
    )

    # Return appropriate output component based on task
    if task == GenerationMode.TextToImage:
        output_component = gr.Image(label="Generated Image", type="pil", visible=True)
        video_component = gr.Video(label="Generated Video", visible=False)
    else:  # TextToVideo
        output_component = gr.Image(label="Generated Image", type="pil", visible=False)
        video_component = gr.Video(label="Generated Video", visible=True)

    # Return updated UI components
    return (
        updated_lora_dropdown,
        updated_subcheckpoint_dropdown,
        updated_resolution_dropdown,
        output_component,
        video_component,
    )


def update_subcheckpoints(checkpoint_dir):
    """Get subdirectories for the selected checkpoint directory."""
    if checkpoint_dir == "None":
        return gr.Dropdown(choices=["None"], value="None", interactive=False, visible=False)

    # Get the full path to the checkpoint directory
    full_checkpoint_path = os.path.join(checkpoint_rootdir, checkpoint_dir)

    # Get all subdirectories
    try:
        subdirs = [
            d
            for d in os.listdir(full_checkpoint_path)
            if os.path.isdir(os.path.join(full_checkpoint_path, d)) and d.startswith("checkpoint-")
        ]
        subdirs.sort()  # Sort to get a consistent order
    except Exception as e:
        logger.error(f"Error loading subdirectories: {str(e)}")
        subdirs = []

    if not subdirs:
        # If there are no subdirectories, hide the dropdown
        return gr.Dropdown(choices=["None"], value="None", interactive=False, visible=False)

    # Show dropdown with available subdirectories
    return gr.Dropdown(
        choices=subdirs,
        label="Checkpoint Version",
        info="Select a specific checkpoint version",
        value=subdirs[-1] if subdirs else None,  # Select the last checkpoint by default
        interactive=True,
        visible=True,
    )


def load_model_and_generate(
    prompt,
    model_type,
    lora_checkpoint,
    subcheckpoint,
    num_inference_steps,
    guidance_scale,
    resolution,
):
    """Load the model with optional LoRA and generate content based on task type."""
    global pipeline, task, prev_model_id

    if not model_type:
        raise gr.Error("Please select a model first")

    if not prompt or prompt.strip() == "":
        raise gr.Error("Please enter a prompt")

    # Create progress tracking
    progress = gr.Progress()
    progress(0, desc="Loading model...")

    # Load the base model
    model_id = "THUDM/" + model_type
    if model_id != prev_model_id:
        prev_model_id = model_id
        pipeline = load_pipeline(
            model_id,
            dtype=torch.bfloat16,
        )

    # Load LoRA weights if selected
    unload_lora_checkpoint(pipeline)
    if lora_checkpoint != "None":
        progress(0.3, desc="Loading LoRA weights...")
        # Construct the full path to the specific checkpoint
        if subcheckpoint and subcheckpoint.strip():
            lora_path = os.path.join(lora_checkpoint, subcheckpoint)
        else:
            lora_path = lora_checkpoint
        logger.info(f"Loading LoRA weights from {lora_path}")
        load_lora_checkpoint(pipeline, lora_path)

    # Generate content based on task
    progress(0.5, desc="Generating content...")

    try:
        if task == GenerationMode.TextToImage:
            height, width = map(int, resolution.split("x"))
            outputs = generate_image(
                prompt=prompt,
                pipeline=pipeline,
                num_images_per_prompt=1,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
            )
            # For image output, return the PIL image and None for video
            return outputs[0], None

        elif task == GenerationMode.TextToVideo:
            frames, height, width = map(int, resolution.split("x"))
            outputs, fps = generate_video(
                prompt=prompt,
                pipeline=pipeline,
                num_videos_per_prompt=1,
                output_type="pil",
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_frames=frames,
                height=height,
                width=width,
            )

            # Create temporary file to save the video
            temp_dir = Path(tempfile.gettempdir()) / "cogkit_videos"
            os.makedirs(temp_dir, exist_ok=True)
            video_path = str(temp_dir / f"{uuid.uuid4()}.mp4")

            # Export video frames to a video file
            export_to_video(outputs[0], video_path, fps=fps)

            # Return None for image and the video path
            return None, video_path

        else:
            raise gr.Error(f"Unsupported task type: {task}")

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise gr.Error(f"Generation failed: {str(e)}")
    finally:
        progress(1.0, desc="Generation completed!")


# ===========================  UI  ===========================

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            with gr.Row():
                model_type = gr.Dropdown(
                    choices=["CogView4-6B", "CogVideoX1.5-5B"],
                    label="Model",
                    info="Select the model to use",
                    interactive=True,
                    value=None,
                )
                lora_dropdown = gr.Dropdown(
                    choices=["None"],
                    label="LoRA Checkpoint",
                    info="Select a LoRA checkpoint to use for inference",
                    interactive=False,
                    value="None",
                )

            subcheckpoint_dropdown = gr.Dropdown(
                choices=[],
                label="Checkpoint Version",
                info="Select a specific checkpoint version",
                interactive=False,
                visible=False,
            )

            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...", lines=3)

            with gr.Row():
                resolution_dropdown = gr.Dropdown(
                    choices=[],
                    label="Resolution",
                    info="Select resolution for generation",
                    interactive=False,
                )

                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Inference Steps",
                    info="Higher values give better quality but take longer",
                )

                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=5.0,
                    step=0.1,
                    label="Guidance Scale",
                    info="Higher values increase prompt adherence",
                )

            generate_btn = gr.Button("Generate")

        with gr.Column(scale=1):
            image_output = gr.Image(label="Generated Image", type="pil")
            video_output = gr.Video(label="Generated Video", visible=False)

    # Set up event handlers
    model_type.change(
        fn=update_task,
        inputs=[model_type],
        outputs=[
            lora_dropdown,
            subcheckpoint_dropdown,
            resolution_dropdown,
            image_output,
            video_output,
        ],
    )

    lora_dropdown.change(
        fn=update_subcheckpoints, inputs=[lora_dropdown], outputs=[subcheckpoint_dropdown]
    )

    generate_btn.click(
        fn=load_model_and_generate,
        inputs=[
            prompt,
            model_type,
            lora_dropdown,
            subcheckpoint_dropdown,
            num_inference_steps,
            guidance_scale,
            resolution_dropdown,
        ],
        outputs=[image_output, video_output],
    )

if __name__ == "__main__":
    demo.launch(share=True, show_error=True)
