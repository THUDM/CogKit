---
---

<!-- TODO: check this doc -->
# Command-Line Interface

`cogkit` provides a powerful command-line interface (CLI) that allows you to perform various tasks without writing Python code. This guide covers the available commands and their usage.

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
```

<!-- TODO: Add example for i2v -->

:::tip
See `cogkit inference --help` for more information.
:::

<!-- TODO: add docs for launch server -->
## Launch Command

The `launch` command will starts a API server:

<!-- FIXME: Add examples -->
```bash
...
```

Please refer to [API](./02-API.md#api-server) for details on how to interact with the API server using client interfaces.

:::tip
See `cogkit launch --help` for more information.
:::
