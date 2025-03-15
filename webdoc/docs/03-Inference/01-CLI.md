---
---

<!-- TODO: check this doc -->
# Command-Line Interface

CogKit provides a powerful command-line interface (CLI) that allows you to perform various tasks without writing Python code. This guide covers the available commands and their usage.

## Overview

The main CLI command is `cogkit`, which provides several subcommands:

```bash
cogkit [OPTIONS] COMMAND [ARGS]...
```

Available commands:
- `inference`: Generate images or videos using AI models
<!-- FIXME: remove this? -->
- `finetune`: Fine-tune models with custom data
- `launch`: Launch a web UI for interactive use

Global options:
- `-v, --verbose`: Increase verbosity (can be used multiple times)

## Inference Command

The `inference` command allows you to generate images and videos:

```bash
cogkit inference [OPTIONS] PROMPT MODEL_ID_OR_PATH
```

### Examples

<!-- FIXME: Add example for i2v -->

```bash
# Generate an image from text
cogkit inference "a beautiful sunset over mountains" "THUDM/CogView4-6B"

# Generate a video from text
cogkit inference "a cat playing with a ball" "THUDM/CogVideoX1.5-5B"

```

<!-- FIXME: remove this? -->
## Fine-tuning Command

The `finetune` command allows you to fine-tune models with your own data:

```bash
cogkit finetune [OPTIONS]
```

> Note: The fine-tuning command is currently under development. Please check back for updates.

<!-- TODO: add docs for launch server -->
## Launch Command

The `launch` command starts a web UI for interactive use:

```bash
cogkit launch [OPTIONS]
```

This launches a web interface where you can:
- Generate images and videos interactively
- Upload images for image-to-video generation
- Adjust generation parameters
- View and download results

### Options

| Option | Description |
|--------|-------------|
| `--host TEXT` | Host to bind the server to (default: 127.0.0.1) |
| `--port INTEGER` | Port to bind the server to (default: 7860) |
| `--share` | Create a public URL |

### Example

```bash
# Launch the web UI on the default port
cogkit launch

```
