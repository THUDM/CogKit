import hashlib
import logging
import math
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from filelock import FileLock
from PIL import Image
from safetensors.torch import load_file, save_file
from torchvision.io import VideoReader

##########  loaders  ##########


def load_prompts(prompt_path: Path) -> list[str]:
    with open(prompt_path, encoding="utf-8") as file:
        return [line.strip() for line in file.readlines() if len(line.strip()) > 0]


def load_videos(video_path: Path) -> list[Path]:
    with open(video_path, encoding="utf-8") as file:
        return [
            video_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images(image_path: Path) -> list[Path]:
    with open(image_path, encoding="utf-8") as file:
        return [
            image_path.parent / line.strip() for line in file.readlines() if len(line.strip()) > 0
        ]


def load_images_from_videos(videos_path: list[Path]) -> list[Path]:
    first_frames_dir = videos_path[0].parent.parent / "first_frames"
    first_frames_dir.mkdir(exist_ok=True)

    first_frame_paths = []
    for video_path in videos_path:
        frame_path = first_frames_dir / f"{video_path.stem}.png"
        if frame_path.exists():
            first_frame_paths.append(frame_path)
            continue

        # Open video
        cap = cv2.VideoCapture(str(video_path))

        # Read first frame
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read video: {video_path}")

        # Save frame as PNG with same name as video
        cv2.imwrite(str(frame_path), frame)
        logging.info(f"Saved first frame to {frame_path}")

        # Release video capture
        cap.release()

        first_frame_paths.append(frame_path)

    return first_frame_paths


##########  preprocessors  ##########


def pil2tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().contiguous()
    return image


def preprocess_image_with_resize(
    image: Image.Image,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Loads and resizes a single image.

    Args:
        image: PIL.Image.Image object
        height: Target height for resizing.
        width: Target width for resizing.
        device: Device to load the data on

    Returns:
        torch.Tensor: Image tensor with shape [C, H, W] where:
            C = number of channels (3 for RGB)
            H = height
            W = width

    Note: The value of returned image tensor should be in the range of 0 ~ 255(rather than 0 ~ 1).
    """
    image = image.convert("RGB")
    image = image.resize((width, height), Image.Resampling.BILINEAR)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().contiguous()

    assert image.shape == (3, height, width)
    return image


def preprocess_video_with_resize(
    video: VideoReader,
    max_num_frames: int,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Loads and resizes a single video.

    The function processes the video through these steps:
      1. If video frame count > max_num_frames, downsample frames evenly
      2. If video dimensions don't match (height, width), resize frames

    Args:
        video: torchvision.io.VideoReader object
        max_num_frames: Maximum number of frames to keep.
        height: Target height for resizing.
        width: Target width for resizing.
        device: Device to load the data on

    Returns:
        A torch.Tensor with shape [F, C, H, W] where:
          F = number of frames
          C = number of channels (3 for RGB)
          H = height
          W = width
    """
    frame_list = list(video)
    video_num_frames = len(frame_list)
    if video_num_frames < max_num_frames:
        # Get all frames first
        frame_torch = torch.stack([f["data"] for f in frame_list])
        # Shape of frame_torch: [F, C, H, W]
        frames = frame_torch.float().to(device)

        # Resize frames to target dimensions
        frames = torch.nn.functional.interpolate(
            frames,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        # Repeat the last frame until we reach max_num_frames
        last_frame = frames[-1:]
        num_repeats = max_num_frames - video_num_frames
        repeated_frames = last_frame.repeat(num_repeats, 1, 1, 1)
        frames = torch.cat([frames, repeated_frames], dim=0)

    else:
        indices = list(range(0, video_num_frames, video_num_frames // max_num_frames))
        # Shape of frames: [F, C, H, W]
        frames = torch.stack([frame_list[i]["data"] for i in indices]).float().to(device)

        # Resize frames to target dimensions
        frames = torch.nn.functional.interpolate(
            frames,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        frames = frames[:max_num_frames]

    return frames.contiguous()


##########  embedding & caching  ##########


def get_prompt_embedding(
    encode_fn: Callable, prompt: str, cache_dir: Path, logger: logging.Logger
) -> torch.Tensor:
    """Get prompt embedding from cache or create new one if not exists.

    Args:
        encode_fn: Function to project prompt to embedding.
        prompt: Text prompt to be embedded
        cache_dir: Base directory for caching embeddings
        logger: Logger instance for logging messages

    Returns:
        torch.Tensor: Prompt embedding with shape [seq_len, hidden_size]
    """
    prompt_embeddings_dir = cache_dir / "prompt_embeddings"
    prompt_embeddings_dir.mkdir(parents=True, exist_ok=True)

    prompt_hash = str(hashlib.sha256(prompt.encode()).hexdigest())
    prompt_embedding_path = prompt_embeddings_dir / (prompt_hash + ".safetensors")
    lock = FileLock(str(prompt_embedding_path) + ".lock")

    with lock:
        if prompt_embedding_path.exists():
            prompt_embedding = load_file(prompt_embedding_path)["prompt_embedding"]
            logger.debug(
                f"Loaded prompt embedding from {prompt_embedding_path}",
                main_process_only=False,
            )
        else:
            prompt_embedding = encode_fn(prompt)
            assert prompt_embedding.ndim == 2
            # shape of prompt_embedding: [seq_len, hidden_size]

            prompt_embedding = prompt_embedding.to("cpu")
            save_file({"prompt_embedding": prompt_embedding}, prompt_embedding_path)
            logger.info(
                f"Saved prompt embedding to {prompt_embedding_path}",
                main_process_only=False,
            )

    return prompt_embedding


def get_image_embedding(
    encode_fn: Callable, image: Image.Image, cache_dir: Path, logger: logging.Logger
) -> torch.Tensor:
    encoded_images_dir = cache_dir / "encoded_images"
    encoded_images_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(image, "filename"):
        logger.warning("Image object does not have filename attribute, skipping caching.")
        return encode_fn(image.convert("RGB")).to("cpu")

    filename = Path(image.filename).stem
    filename_hash = str(hashlib.sha256(filename.encode()).hexdigest())
    encoded_image_path = encoded_images_dir / (filename_hash + ".safetensors")

    if encoded_image_path.exists():
        encoded_image = load_file(encoded_image_path)["encoded_image"]
        logger.debug(
            f"Loaded encoded image from {encoded_image_path}",
            main_process_only=False,
        )
    else:
        encoded_image = encode_fn(image.convert("RGB"))
        encoded_image = encoded_image.to("cpu")
        save_file({"encoded_image": encoded_image}, encoded_image_path)
        logger.info(
            f"Saved encoded image to {encoded_image_path}",
            main_process_only=False,
        )

    return encoded_image


def calculate_resize_dimensions(height: int, width: int, max_pixels: int) -> tuple[int, int]:
    """
    Calculate new dimensions for an image while maintaining aspect ratio and limiting total pixels.

    Args:
        height (int): Original height of the image
        width (int): Original width of the image
        max_pixels (int): Maximum number of pixels allowed

    Returns:
        Tuple[int, int]: New (width, height) dimensions
    """
    current_pixels = width * height

    # If current pixel count is already below max, return original dimensions
    if current_pixels <= max_pixels:
        return height, width

    # Calculate scaling factor to maintain aspect ratio
    scale = math.sqrt(max_pixels / current_pixels)

    # Calculate new dimensions
    new_height = int(height * scale)
    new_width = int(width * scale)

    return new_height, new_width
