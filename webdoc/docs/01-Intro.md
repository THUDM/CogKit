---
slug: /
---

# Introduction

CogKit is an open-source project that provides a user-friendly interface for researchers and developers to utilize ZhipuAI's [**CogView**](https://huggingface.co/collections/THUDM/cogview-67ac3f241eefad2af015669b) (image generation) and [**CogVideoX**](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) (video generation) models. It streamlines multimodal tasks such as **text-to-image (T2I)**, **text-to-video (T2V)**, and **image-to-video (I2V)**. Users must comply with legal and ethical guidelines to ensure responsible implementation.

## Supported Models

Please refer to the [Model Card](./05-Model%20Card.mdx) for more details.

## Environment Testing

This repository has been tested in environments with `1×A100` and `8×A100` GPUs, using `CUDA 12.4, Python 3.10.16`.

- Cog series models typically do not support `FP16` precision (Only `CogVideoX-2B` support); GPUs like the `V100` cannot be fine-tuned properly (Will cause `loss=nan` for example). At a minimum, an `A100` or other GPUs supporting `BF16` precision should be used.
- We have not yet systematically tested the minimum GPU memory requirements for each model. For `LORA(bs=1 with offload)`, a single `A100` GPU is sufficient. For `SFT`, our tests have passed in an `8×A100` environment.
