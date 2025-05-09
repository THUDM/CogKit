#! /usr/bin/env python

import argparse

import torch
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from pathlib import Path
from cogkit.utils.lora import _LORA_WEIGHT_NAME
from safetensors.torch import save_file

TORCH_SAVE_CHECKPOINT_DIR = "diffusion_pytorch_model.bin"


def main(checkpoint_dir: str, output_dir: str, is_lora: bool = False):
    # convert dcp model to torch.save (assumes checkpoint was generated as above)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = output_dir / TORCH_SAVE_CHECKPOINT_DIR

    print("Converting FSDP checkpoint to torch.save format...")
    dcp_to_torch_save(checkpoint_dir, ckpt_file)
    state = torch.load(ckpt_file, map_location="cpu")
    print("Deleting torch checkpoint...")
    ckpt_file.unlink()
    model_weights = state["app"]["model"]

    print("Saving transformer weights...")
    if is_lora:
        ckpt_file = ckpt_file.with_name(_LORA_WEIGHT_NAME)
        save_file(model_weights, ckpt_file)

    else:
        ckpt_file = ckpt_file.with_name(TORCH_SAVE_CHECKPOINT_DIR)
        torch.save(model_weights, ckpt_file)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--lora", action="store_true", default=False)
    args = parser.parse_args()

    main(args.checkpoint_dir, args.output_dir, args.lora)
