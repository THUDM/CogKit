---
---

# Installation

CogKit can be installed using pip. We recommend using a virtual environment to avoid conflicts with other packages.

## Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (for optimal performance)
- At least 8GB of GPU memory for inference, 16GB+ recommended for fine-tuning

## Installation Steps

### Create a virtual environment (recommended)

```bash
# Using venv
python -m venv cogkit-env
source cogkit-env/bin/activate

# Or using conda
conda create -n cogkit-env python=3.8
conda activate cogkit-env
```

### Install PyTorch

Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch according to your system.

### Install Cogkit
<!-- FIXME: Install via pip install cogkit or via clone&local install? -->

1. Install Cogkit:

   <!-- TODO: add github link -->
   ```bash
   pip install cogkit@git+https:
   ```

2. Optional: for video tasks (e.g. text-to-video), install additional dependencies:

   ```bash
   pip install -e .[video]
   ```


### Verify installation

You can verify that cogkit is installed correctly by running:

```bash
python -c "import cogkit"
```

<!-- TODO: add in roadmap -->
## [Optional] Install via docker

If you have any issues with the installation, you can install Cogkit via Docker. We provide a Docker image that includes all dependencies. You can pull the image from Docker Hub:

<!-- FIXME: add link to the docker image -->
```bash
docker pull ghcr.io/cogkit/cogkit:latest
```

To run the container, use the following command:

<!-- FIXME: add link to the docker image -->
```bash
docker run -it ghcr.io/cogkit/cogkit:latest
```

## Troubleshooting

For more detailed troubleshooting, please refer to our GitHub issues page.
