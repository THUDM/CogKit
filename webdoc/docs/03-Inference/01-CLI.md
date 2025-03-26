---
---

# Command-Line Interface

CogKit provides a powerful command-line interface (CLI) that allows you to perform various tasks without writing Python code. This guide covers the available commands and their usage.

## Overview

The main CLI command is `cogkit`, which provides several subcommands:

```bash
cogkit [OPTIONS] COMMAND [ARGS]...
```

Available commands:

- `inference`: Generate images or videos using AI models
- `launch`: Launch a API server

Global options:

- `-v, --verbose`: Increase verbosity (can be used multiple times)

## Inference Command

The `inference` command allows you to generate images or videos:


```bash
# Generate an image from text
cogkit inference "a beautiful sunset over mountains" "THUDM/CogView4-6B"

# Generate a video from text
cogkit inference "a cat playing with a ball" "THUDM/CogVideoX1.5-5B"

# Controlling image generation from text
cogkit inference "a dog" "THUDM/CogView4-6B-Control"
```

:::tip
See `cogkit inference --help` for more information.
:::

## Launch Command

The `launch` command starts an API server for image and video generation. Before using this command, you need to install the API dependencies:

<!-- FIXME: check url -->
```bash
pip install "cogkit[api]@git+https://github.com/THUDM/CogKit.git"
```

<!-- FIXME: check url -->
Before starting the server, make sure to configure the model paths that you want to serve. This step is necessary to specify which models will be available through the API server.

To configure the model paths:

1. Create a `.env` file in your working directory
2. Refer to the [environment template](https://github.com/THUDM/CogKit/blob/main/.env.template) and add needed environment variables to specify model paths. For example, to serve `CogView4-6B` as a service, you must specify `COGVIEW4_PATH` in your `.env` file:

    ```bash
    # /your/workdir/.env

    COGVIEW4_PATH="THUDM/CogView4-6B"  # or local path
    # other variables...
    ```

Then starts a API server, for example:

```bash
cogkit launch
```

:::tip
See `cogkit launch --help` for more information.
:::


### Client Interfaces

The server API is OpenAI-compatible, which means you can use it with any OpenAI client library. Here's an example using the OpenAI Python client:

```python
import base64

from io import BytesIO
from PIL import Image

from openai import OpenAI

client = OpenAI(
    api_key="foo",
    base_url="http://localhost:8000/v1"  # Your server URL
)

# Generate an image from cogview-4
response = client.images.generate(
    model="cogview-4",
    prompt="a beautiful sunset over mountains",
    n=1,
    size="1024x1024",
)
image_b64 = response.data[0].b64_json

# Decode the base64 string
image_data = base64.b64decode(image_b64)

# Create an image from the decoded data
image = Image.open(BytesIO(image_data))

# Save the image
image.save("output.png")
```
