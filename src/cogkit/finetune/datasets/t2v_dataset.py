from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from safetensors.torch import load_file, save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import VideoReader
from typing_extensions import override

from cogkit.finetune.diffusion.constants import LOG_LEVEL, LOG_NAME

from .utils import get_prompt_embedding, preprocess_video_with_resize

if TYPE_CHECKING:
    from cogkit.finetune.diffusion.trainer import DiffusionTrainer

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseT2VDataset(Dataset):
    """
    Base dataset class for Text-to-Video (T2V) training.

    This dataset loads prompts and videos for T2V training.

    Args:
        data_root (str): Root directory containing the dataset files
        device (torch.device): Device to load the data on
        trainer (DiffusionTrainer): Trainer object
        using_train (bool): Whether to use the training set
    """

    def __init__(
        self,
        data_root: str,
        device: torch.device = None,
        trainer: "DiffusionTrainer" = None,
        using_train: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        self.using_train = using_train
        self.data_root = Path(data_root)
        if using_train:
            self.data_root = self.data_root / "train"
            self.data = load_dataset("videofolder", data_dir=self.data_root, split="train")
        else:
            self.data_root = self.data_root / "test"
            self.data = load_dataset("json", data_dir=self.data_root, split="train")

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
        prompt_embedding = get_prompt_embedding(self.encode_text, prompt, cache_dir, logger)

        if not self.using_train:
            return {
                "prompt": prompt,
                "prompt_embedding": prompt_embedding,
            }

        ##### video
        video = self.data[index]["video"]
        video_path = Path(video._hf_encoded["path"])

        train_resolution_str = "x".join(str(x) for x in self.trainer.args.train_resolution)

        video_latent_dir = (
            cache_dir / "video_latent" / self.trainer.args.model_name / train_resolution_str
        )
        video_latent_dir.mkdir(parents=True, exist_ok=True)
        encoded_video_path = video_latent_dir / (video_path.stem + ".safetensors")

        if encoded_video_path.exists():
            encoded_video = load_file(encoded_video_path)["encoded_video"]
            logger.debug(
                f"Loaded encoded video from {encoded_video_path}",
                main_process_only=False,
            )
        else:
            frames = self.preprocess(video, self.device)
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
            logger.info(
                f"Saved encoded video to {encoded_video_path}",
                main_process_only=False,
            )

        return {
            "prompt": prompt,
            "prompt_embedding": prompt_embedding,
            "encoded_video": encoded_video,  # shape: [C, F, H, W]
        }

    def preprocess(
        self, video: VideoReader, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """
        Loads and preprocesses a video.

        Args:
            video: torchvision.io.VideoReader object
            device: torch.device

        Returns:
            torch.Tensor: Video tensor of shape [F, C, H, W] where:
                - F is number of frames
                - C is number of channels (3 for RGB)
                - H is height
                - W is width
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
            torch.Tensor: The transformed video tensor with the same shape as the input
        """
        raise NotImplementedError("Subclass must implement this method")


class T2VDatasetWithResize(BaseT2VDataset):
    """
    A dataset class for text-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width

    Args:
        max_num_frames (int): Maximum number of frames to extract from videos
        height (int): Target height for resizing videos
        width (int): Target width for resizing videos
    """

    def __init__(self, train_resolution: tuple[int, int, int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.max_num_frames = train_resolution[0]
        self.height = train_resolution[1]
        self.width = train_resolution[2]

        self.__frame_transform = transforms.Compose(
            [transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0)]
        )

    @override
    def preprocess(
        self, video: VideoReader, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        return preprocess_video_with_resize(
            video,
            self.max_num_frames,
            self.height,
            self.width,
            device,
        )

    @override
    def video_transform(self, frames: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.__frame_transform(f) for f in frames], dim=0)
