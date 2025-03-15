---
---

# API

CogKit provides a powerful inference API for generating images and videos using various AI models. This document covers both the Python API and API server.

## Python API

You can also use Cogkit programmatically in your Python code:

```python
from cogkit.generation import generate_image, generate_video

# Text-to-Image generation
image = generate_image(
    prompt="a beautiful sunset over mountains",
    model_id_or_path="THUDM/CogView4-6B",
    lora_model_id_or_path=None,
    transformer_path=None,
)
image.save("sunset.png")

# Text-to-Video generation
video = generate_video(
    prompt="a cat playing with a ball",
    model_id_or_path="THUDM/CogVideoX1.5-5B",
    lora_model_id_or_path=None,
    transformer_path=None,
    num_frames=81,
    fps=16,
)
video.save("cat_video.mp4")
```

## API Server

<!-- FIXME: add docs for the API server -->

<!-- TODO: add examples -->
