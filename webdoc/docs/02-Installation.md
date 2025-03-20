---
---

# Installation

`cogkit` can be installed using pip. We recommend using a virtual environment to avoid conflicts with other packages.

## Requirements

- Python 3.10 or higher
- OpenCV and PyTorch

## Installation Steps

### OpenCV

Please refer to the [opencv-python installation guide](https://github.com/opencv/opencv-python?tab=readme-ov-file#installation-and-usage) for instructions on installing OpenCV according to your system.

### PyTorch

Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch according to your system.

### CogKit

1. Install `cogkit`:

   ```bash
   pip install cogkit@git+https://github.com/THUDM/cogkit.git
   ```

2. Optional: for video tasks (e.g. text-to-video), install additional dependencies:

   ```bash
   pip install -e .[video]
   ```

### Verify installation

You can verify that cogkit is installed correctly by running:

```bash
cogkit --help
```

and will get:

```text
Usage: cogkit [OPTIONS] COMMAND [ARGS]...

Options:
  -v, --verbose  Verbosity level (from 0 to 2)  [default: 0; 0<=x<=2]
  --help         Show this message and exit.

Commands:
  finetune
  inference  Generates a video based on the given prompt and saves it to...
  launch
```
