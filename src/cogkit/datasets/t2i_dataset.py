import torchvision.transforms as transforms
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple
from PIL import Image

import torch
from accelerate.logging import get_logger
from datasets import load_dataset
from torch.utils.data import Dataset
from typing_extensions import override

from cogkit.finetune.diffusion.constants import LOG_LEVEL, LOG_NAME

from .utils import (
    preprocess_image_with_resize,
    get_prompt_embedding,
    get_image_embedding,
    pil2tensor,
)

if TYPE_CHECKING:
    from cogkit.finetune.diffusion.trainer import DiffusionTrainer

logger = get_logger(LOG_NAME, LOG_LEVEL)


class BaseT2IDataset(Dataset):
    """
    Base dataset class for Text-to-Image (T2I) training.

    This dataset loads prompts and corresponding conditioning images for T2I training.

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
            self.data = load_dataset("imagefolder", data_dir=self.data_root, split="train")

        else:
            self.data_root = self.data_root / "test"
            self.data = load_dataset("json", data_dir=self.data_root, split="train")

        self.device = device
        self.encode_text = trainer.encode_text
        self.encode_image = trainer.encode_image
        self.trainer = trainer

        self._image_transforms = transforms.Compose(
            [
                transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
            ]
        )

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

        ##### image
        image = self.data[index]["image"]
        image_original: Image.Image = image

        def encode_fn(image: Image.Image) -> torch.Tensor:
            image_preprocessed = self.preprocess(image, self.device)
            image_preprocessed = self.image_transform(image_preprocessed)
            encoded_image = self.trainer.encode_image(image_preprocessed[None, ...])[0]

            return encoded_image

        # shape of encoded_image: [C, H, W]
        encoded_image = get_image_embedding(encode_fn, image, cache_dir, logger)

        # shape of image: [C, H, W]
        return {
            "image": image_original,
            "encoded_image": encoded_image,
            "prompt": prompt,
            "prompt_embedding": prompt_embedding,
        }

    def preprocess(
        self,
        image: Image.Image,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Loads and preprocesses an image.

        Args:
            image: PIL.Image.Image object
            device: Device to load the data on

        Returns:
            - image(torch.Tensor) of shape [C, H, W]
        """
        return pil2tensor(image)

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
        return self._image_transforms(image)


class T2IDatasetWithResize(BaseT2IDataset):
    """
    A dataset class for image-to-video generation that resizes inputs to fixed dimensions.

    This class preprocesses videos and images by resizing them to specified dimensions:
    - Videos are resized to max_num_frames x height x width
    - Images are resized to height x width

    Args:
        height (int): Target height for resizing videos and images
        width (int): Target width for resizing videos and images
    """

    def __init__(self, train_resolution: Tuple[int, int], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.height = train_resolution[0]
        self.width = train_resolution[1]

    @override
    def preprocess(
        self,
        image: Image.Image,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        image = preprocess_image_with_resize(image, self.height, self.width, device)
        return image


class T2IDatasetWithPacking(Dataset):
    """
    This dataset class packs multiple samples from a base Text-to-Image dataset.
    It should be used combined with PackingSampler.
    """

    def __init__(
        self,
        base_dataset: BaseT2IDataset,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert type(base_dataset) is BaseT2IDataset  # should literally be a BaseT2IDataset
        self.base_dataset = base_dataset

    def __getitem__(self, index: list[int]) -> dict[str, Any]:
        return {
            "image": [self.base_dataset[i]["image"] for i in index],
            "encoded_image": [self.base_dataset[i]["encoded_image"] for i in index],
            "prompt": [self.base_dataset[i]["prompt"] for i in index],
            "prompt_embedding": [self.base_dataset[i]["prompt_embedding"] for i in index],
        }

    def __len__(self) -> int:
        return len(self.base_dataset)
