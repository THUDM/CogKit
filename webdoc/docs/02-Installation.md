---
---

# Installation

CogKit can be installed using pip. We recommend using a virtual environment to avoid conflicts with other packages.

## Requirements

- Python 3.10 or higher
- CUDA-compatible GPU (for optimal performance)
- At least 8GB of GPU memory for inference, 16GB+ recommended for fine-tuning

## Installation Steps

### Create a virtual environment (recommended)

```bash
# Using venv
python -m venv cogkit-env
source cogkit-env/bin/activate

# Or using conda
conda create -n cogkit-env python=3.10
conda activate cogkit-env
```

### Install PyTorch

Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch according to your system.

### Install Cogkit

1. Install Cogkit:
   ```bash
   pip install cogkit@git+https://github.com/thudm/cogkit.git
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
