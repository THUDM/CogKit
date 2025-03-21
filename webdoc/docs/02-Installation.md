---
---

# Installation

## Requirements

- Python 3.10 or higher
- PyTorch, OpenCV, decord

## Installation Steps

### PyTorch

Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch according to your system.

### OpenCV

Please refer to the [OpenCV installation guide](https://github.com/opencv/opencv-python?tab=readme-ov-file#installation-and-usage) to install opencv-python. In most cases, you can simply install by `pip install opencv-python-headless`

### decord

Please refer to the [decord installation guide](https://github.com/dmlc/decord?tab=readme-ov-file#installation) to install decord dependencies. If you don't need GPU acceleration, you can simply install by `pip install decord`

### CogKit

Install `cogkit` from github source:

```bash
pip install "cogkit@git+https://github.com/THUDM/cogkit.git"
```


### Verify installation

You can verify that cogkit is installed correctly by running:

```bash
cogkit --help
```
