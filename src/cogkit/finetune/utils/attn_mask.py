import math
from typing import Any, List, Tuple

import torch
from transformers import AutoTokenizer
from diffusers.models.attention_processor import Attention

from .filters import MeanFilter


def mask_assert(mask: torch.Tensor) -> None:
    assert torch.all((mask == 0) | (mask == 1)), "mask contains values other than 0 or 1"
    assert mask.dtype == torch.int32, "mask dtype should be torch.int32"


def process_prompt_attention_mask(
    tokenizer: AutoTokenizer,
    prompt: List[str],
    prompt_embedding: List[torch.Tensor],
    max_ttoken_length: int,
    pad_to_multiple_of: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process prompt attention mask for training.

    Args:
        tokenizer: HuggingFace AutoTokenizer instance
        prompt: List of text prompts
        prompt_embedding: List of prompt embeddings
        max_ttoken_length: Maximum text token length

    Returns:
        A tuple of (prompt_embedding, prompt_attention_mask)

    Attention mask values:
        0: Padding tokens (tokens to be ignored)
        1: True tokens (tokens to be attended)
    """
    # Tokenize the prompt in the batch-level
    tokenized_prompt = tokenizer(
        prompt,
        padding="longest",
        max_length=max_ttoken_length,
        truncation=True,
        add_special_tokens=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
    )
    prompt_attention_mask = tokenized_prompt.attention_mask
    num_samples = len(prompt)

    token_length_list = [embedding.shape[0] for embedding in prompt_embedding]
    max_seqlen = max(token_length_list)
    assert max_seqlen == prompt_attention_mask.shape[1]

    prompt_embedding = torch.nn.utils.rnn.pad_sequence(
        prompt_embedding, batch_first=True, padding_value=0, padding_side="left"
    )

    assert prompt_embedding.shape[0] == prompt_attention_mask.shape[0] == num_samples
    assert prompt_embedding.shape[1] == prompt_attention_mask.shape[1] == max_seqlen

    prompt_attention_mask = prompt_attention_mask.to(torch.int32)
    mask_assert(prompt_attention_mask)

    return prompt_embedding, prompt_attention_mask


def process_latent_attention_mask(
    encoded_images: List[torch.Tensor], patch_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process latent attention mask for training.

    Args:
        encoded_images: List of encoded image latents with shape [num_channels, height, width]
                        where each image might have different height and width
        patch_size: Patch size for the transformer

    Returns:
        A tuple of (padded_latent, vtoken_attention_mask, pixel_mask)

    Attention mask values:
        0: Padding tokens (tokens to be ignored)
        1: True tokens (tokens to be attended)
    """
    num_samples = len(encoded_images)
    num_latent_channel = encoded_images[0].shape[0]

    # Calculate maximum dimensions for padding
    max_latent_height = max([img.shape[1] for img in encoded_images])
    max_latent_width = max([img.shape[2] for img in encoded_images])

    # Ensure dimensions are divisible by patch_size
    max_latent_height = math.ceil(max_latent_height / patch_size) * patch_size
    max_latent_width = math.ceil(max_latent_width / patch_size) * patch_size

    # Create padded latent tensor and pixel mask
    padded_latent = torch.zeros(
        num_samples,
        num_latent_channel,
        max_latent_height,
        max_latent_width,
        dtype=torch.float32,
    )
    pixel_mask = padded_latent.clone()

    # Fill padded latent and set mask values
    for idx, latent in enumerate(encoded_images):
        padded_latent[idx, :, : latent.shape[1], : latent.shape[2]] = latent
        pixel_mask[idx, :, : latent.shape[1], : latent.shape[2]] = 1

    # Ensure dimensions are divisible by patch_size
    assert max_latent_height % patch_size == 0 and max_latent_width % patch_size == 0

    # Apply mean filter to create vtoken attention mask
    mean_filter = MeanFilter(kernel_size=patch_size, in_channels=num_latent_channel)
    vtoken_attention_mask = mean_filter(pixel_mask).squeeze(1)  # remove channel dimension

    vtoken_attention_mask[vtoken_attention_mask != 1] = 0

    pixel_mask = pixel_mask.to(torch.int32)
    vtoken_attention_mask = vtoken_attention_mask.to(torch.int32)
    mask_assert(pixel_mask)
    mask_assert(vtoken_attention_mask)

    return padded_latent, vtoken_attention_mask, pixel_mask


def replace_attn_processor(model: torch.nn.Module, attn_processor_obj: Any) -> None:
    for name, submodule in model.named_modules():
        if isinstance(submodule, Attention):
            submodule.processor = attn_processor_obj
