import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Tuple

import torch
import torchvision.transforms as transforms
from datasets import load_dataset
from PIL import Image
from torch.utils.data import Dataset
from typing_extensions import override

from cogkit.finetune.logger import get_logger

from .utils import (
    calculate_resize_dimensions,
    get_image_embedding,
    get_prompt_embedding,
    pil2tensor,
    preprocess_image_with_resize,
)

if TYPE_CHECKING:
    from cogkit.finetune.diffusion.trainer import DiffusionTrainer

_logger = get_logger()


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
        prompt_embedding = get_prompt_embedding(self.encode_text, prompt, cache_dir)

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
        encoded_image = get_image_embedding(encode_fn, image, cache_dir)

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

        **Note**: The value of returned image tensor should be the float value in the range of 0 ~ 255(rather than 0 ~ 1).
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


class T2IDatasetWithFactorResize(BaseT2IDataset):
    """
    A dataset class that resizes images to dimensions that are multiples of a specified factor.

    If the image dimensions are not divisible by the factor, the image is resized
    to the nearest larger dimensions that are divisible by the factor.

    Args:
        factor (int): The factor that image dimensions should be divisible by
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.factor = self.trainer.IMAGE_FACTOR

    @override
    def preprocess(
        self,
        image: Image.Image,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Preprocesses an image by resizing it to dimensions that are multiples of self.factor.

        Args:
            image: PIL.Image.Image object
            device: Device to load the data on

        Returns:
            torch.Tensor: Processed image tensor of shape [C, H, W]
        """
        # Get original dimensions
        width, height = image.size
        maxpixels = self.trainer.state.train_resolution[0] * self.trainer.state.train_resolution[1]
        new_height, new_width = calculate_resize_dimensions(height, width, maxpixels)

        # Calculate nearest multiples of factor (rounding down)
        new_height = math.floor(new_height / self.factor) * self.factor
        new_width = math.floor(new_width / self.factor) * self.factor

        assert new_height > 0 and new_width > 0, "Have image with height or width <= self.factor"

        return preprocess_image_with_resize(image, new_height, new_width, device)


class T2IDatasetWithPacking(Dataset):
    """
    This dataset class packs multiple samples from a base Text-to-Image dataset.
    It should be used combined with PackingSampler.
    """

    def __init__(
        self,
        base_dataset: T2IDatasetWithFactorResize,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # base_dataset should be a T2IDatasetWithFactorResize
        assert type(base_dataset) is T2IDatasetWithFactorResize

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
