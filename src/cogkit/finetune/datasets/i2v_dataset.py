# -*- coding: utf-8 -*-


from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import VideoReader
from typing_extensions import override

from cogkit.finetune.logger import get_logger

from .utils import (
    get_prompt_embedding,
    preprocess_image_with_resize,
    preprocess_video_with_resize,
)

if TYPE_CHECKING:
    from cogkit.finetune.diffusion.trainer import DiffusionTrainer

_logger = get_logger()


class BaseI2VDataset(Dataset):
    """
    Base dataset class for Image-to-Video (I2V) training.

    This dataset loads prompts, videos and corresponding conditioning images for I2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        device (torch.device): Device to load the data on
        trainer (DiffusionTrainer): Trainer object
        using_train (bool): Whether to use the training set
    """

    def __init__(
        self,
        data_root: str,
        device: torch.device,
        trainer: "DiffusionTrainer" = None,
        using_train: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.data_root = Path(data_root)
        self.using_train = using_train

        if using_train:
            self.data_root = self.data_root / "train"
            metadata_path = self.data_root / "metadata.jsonl"
            video_path = self.data_root / "videos"
            image_path = self.data_root / "images"

            metadata = load_dataset("json", data_files=str(metadata_path), split="train")
            video_data = load_dataset("videofolder", data_dir=video_path, split="train")
            metadata = metadata.sort("id")
            video_data = video_data.sort("id")

            def update_with_prompt(video_example, idx):
                video_example["prompt"] = metadata[idx]["prompt"]
                return video_example

            video_data = video_data.map(update_with_prompt, with_indices=True)

            if image_path.exists():
                image_data = load_dataset("imagefolder", data_dir=image_path, split="train")
                image_data = image_data.sort("id")

                # Map function to update video dataset with corresponding images
                def update_with_image(video_example, idx):
                    video_example["image"] = image_data[idx]["image"]
                    return video_example

                self.data = video_data.map(update_with_image, with_indices=True)

            else:
                _logger.warning(
                    f"No image data found in {self.data_root}, using first frame of video instead"
                )

                def add_first_frame(example):
                    assert len(example["video"]) == 1
                    video: VideoReader = example["video"][0]
                    # shape of first_frame: [C, H, W]
                    first_frame = next(video)["data"]
                    to_pil = transforms.ToPILImage()
                    example["image"] = [to_pil(first_frame)]
                    return example

                self.data = video_data.with_transform(add_first_frame)

        else:
            self.data_root = self.data_root / "test"
            self.data = load_dataset("imagefolder", data_dir=self.data_root, split="train")

        self.device = device
        self.encode_video = trainer.encode_video
        self.encode_text = trainer.encode_text
        self.trainer = trainer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cache_dir = self.data_root / ".cache"

        ##### prompt
        prompt = self.data[index]["prompt"]
        prompt_embedding = get_prompt_embedding(self.encode_text, prompt, cache_dir)

        ##### image
        image_preprocessed = self.data[index]["image"]
        image_original: Image.Image = image_preprocessed
        _, image_preprocessed = self.preprocess(None, image_preprocessed, self.device)
        image_preprocessed = self.image_transform(image_preprocessed)
        image_preprocessed = image_preprocessed.to("cpu")
        # shape of image: [C, H, W]

        if not self.using_train:
            return {
                "image": image_original,
                "image_preprocessed": image_preprocessed,
                "prompt": prompt,
                "prompt_embedding": prompt_embedding,
            }

        ##### video
        video = self.data[index]["video"]
        video_path = Path(video._hf_encoded["path"])
        train_resolution_str = "x".join(str(x) for x in self.trainer.uargs.train_resolution)

        video_latent_dir = (
            cache_dir / "video_latent" / self.trainer.uargs.model_name / train_resolution_str
        )
        video_latent_dir.mkdir(parents=True, exist_ok=True)

        encoded_video_path = video_latent_dir / (video_path.stem + ".safetensors")

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            _logger.debug(f"Loaded encoded video from {encoded_video_path}")
        else:
            frames, _ = self.preprocess(video, None, self.device)
            # Current shape of frames: [F, C, H, W]
            frames = self.video_transform(frames)
            # Convert to [B, C, F, H, W]
            frames = frames.unsqueeze(0)
            frames = frames.permute(0, 2, 1, 3, 4).contiguous()
            encoded_video = self.encode_video(frames)

            # [1, C, F, H, W] -> [C, F, H, W]
            encoded_video = encoded_video[0]
            encoded_video = encoded_video.to("cpu")
            save_file({"encoded_video": encoded_video}, encoded_video_path)
            _logger.info(f"Saved encoded video to {encoded_video_path}")

        # shape of encoded_video: [C, F, H, W]
        # shape of image: [C, H, W]
        return {
            "image": image_original,
            "image_preprocessed": image_preprocessed,
            "prompt": prompt,
            "prompt_embedding": prompt_embedding,
            "video": video,
            "encoded_video": encoded_video,
        }

    def preprocess(
        self,
        video: VideoReader | None,
        image: Image.Image | None,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Loads and preprocesses a video and an image.
        If either path is None, no preprocessing will be done for that input.

        Args:
            video: torchvision.io.VideoReader object
            image: PIL.Image.Image object
            device: Device to load the data on

        Returns:
            A tuple containing:
                - video(torch.Tensor) of shape [F, C, H, W] where F is number of frames,
                  C is number of channels, H is height and W is width
                - image(torch.Tensor) of shape [C, H, W]
        """
        raise NotImplementedError("Subclass must implement this method")

    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to a video.

        Args:
            frames (torch.Tensor): A 4D tensor representing a video
                with shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed video tensor
        """
        raise NotImplementedError("Subclass must implement this method")

    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        """
        Applies transformations to an image.

        Args:
            image (torch.Tensor): A 3D tensor representing an image
                with shape [C, H, W] where:
                - C is number of channels (3 for RGB)
                - H is height
                - W is width

        Returns:
            torch.Tensor: The transformed image tensor
        """
        raise NotImplementedError("Subclass must implement this method")


class I2VDatasetWithResize(BaseI2VDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    # def __init__(self, max_num_frames: int, height: int, width: int, *args, **kwargs) -> None:
    def __init__(self, train_resolution: Tuple[int, int, int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = train_resolution[0]
        self.height = train_resolution[1]
        self.width = train_resolution[2]

        self.__frame_transforms = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )
        self.__image_transforms = self.__frame_transforms

    @override
    def preprocess(
        self,
        video: VideoReader | None,
        image: Image.Image | None,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video is not None:
            video = preprocess_video_with_resize(
                video, self.max_num_frames, self.height, self.width, device
            )
        else:
            video = None
        if image is not None:
            image = preprocess_image_with_resize(image, self.height, self.width, device)
        else:
            image = None
        return video, image

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transforms(f) for f in frames], dim=0)

    @override
    def image_transform(self, image: torch.Tensor) -> torch.Tensor:
        return self.__image_transforms(image)
