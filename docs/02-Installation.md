---
---

# Installation

## Requirements

- Python 3.10 or higher
- PyTorch

## Installation Steps

### PyTorch

Please refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions on installing PyTorch according to your system.

### CogKit

1. Install `cogkit`:

   ```bash
   pip install "cogkit@git+https://github.com/THUDM/cogkit.git"
   ```

2. Optional: for video tasks (e.g. text-to-video), install additional dependencies:

   ```bash
   pip install "cogkit[video]@git+https://github.com/THUDM/cogkit.git"
   ```

### Verify installation

You can verify that cogkit is installed correctly by running:

```bash
cogkit --help
```
