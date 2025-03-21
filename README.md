# CogKit

## Introduction

CogKit is an open-source project that provides a user-friendly interface for researchers and developers to utilize ZhipuAI's [**CogView**](https://huggingface.co/collections/THUDM/cogview-67ac3f241eefad2af015669b) (image generation) and [**CogVideoX**](https://huggingface.co/collections/THUDM/cogvideo-66c08e62f1685a3ade464cce) (video generation) models. It streamlines multimodal tasks such as **text-to-image (T2I)**, **text-to-video (T2V)**, and **image-to-video (I2V)**. Users must comply with legal and ethical guidelines to ensure responsible implementation.

Visit our [**Docs**](https://thudm.github.io/CogKit) to start.

## Features

- **Fine-tuning Methods**: Supports **LoRA** and **full-parameter fine-tuning** across various setups, including **single-machine single-GPU**, **single-machine multi-GPU**, and **multi-machine multi-GPU** configurations.
- **Inference**: Provides an **OpenAI-style API** (T2I Only) and a **command-line interface** for seamless model deployment.
- **Embed Cache**: Optimizes GPU memory usage to enhance efficiency during inference.

## Roadmap

- [ ] Add support for CogView4 ControlNet model
- [ ] Docker for easy deployment

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
