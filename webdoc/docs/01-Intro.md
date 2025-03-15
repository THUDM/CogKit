---
slug: /
---

# Introduction

CogKit is a powerful framework for working with ZhipuAI Cog Series models, focusing on multimodal generation and fine-tuning capabilities. 
It provides a unified interface for various AI tasks including text-to-image, text-to-video, and image-to-video generation.

## Key Features

- **Command-line Interface**: Easy-to-use CLI and Python API for both inference and fine-tuning
- **Fine-tuning Support**: With LoRA or full model fine-tuning support to customize models with your own data

## Supported Models

Please refer to the [Model Card](./05-Model%20Card.mdx) for more details.

## Environment Testing

This repository has been tested in environments with `1×A100` and `8×A100` GPUs, using `CUDA 12.4, Python 3.10.16`.

- Cog series models typically do not support `FP16` precision (Only `CogVideoX-2B` support); GPUs like the `V100` cannot be fine-tuned properly (Will cause `loss=nan` for example). At a minimum, an `A100` or other GPUs supporting `BF16` precision should be used.
- We have not yet systematically tested the minimum GPU memory requirements for each model. For `LORA(bs=1 with offload)`, a single `A100` GPU is sufficient. For `SFT`, our tests have passed in an `8×A100` environment.

## Roadmap

- [ ] Add support for CogView4 ControlNet model
- [ ] Docker Image for easy deployment
- [ ] Embedding Cache to Reduce GPU Memory Usage