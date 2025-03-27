---
---

# Dataset Format

`CogKit/quickstart/data` directory contains various dataset templates for fine-tuning different models, please refer to the corresponding dataset template based on your task type:

## Text-to-Image Conversion Dataset (t2i)

- Each directory contains a set of image files (`.png`)
- The `metadata.jsonl` file contains text descriptions for each image

    ```json
    {"file_name": "example.png", "prompt": "Detailed image description text..."}
    ```

## Text-to-Video (t2v)

- Each directory contains a set of video files (`.mp4`)
- The `metadata.jsonl` file contains text descriptions for each video

    ```json
    {"file_name": "example.mp4", "prompt": "Detailed video description text..."}
    ```

## Image-to-Video (i2v)

- The dataset is organized with the following structure:
  - `train/` and `test/` directories each containing:
    - `videos/` directory for video files (`.mp4`)
    - `images/` directory for input image files (`.png`)
    - `metadata.jsonl` file in the root containing prompt descriptions

- The main `metadata.jsonl` file in the root directory contains prompt information for each sample:
  ```json
  {"id": 0, "prompt": "Detailed video description text..."}
  {"id": 1, "prompt": "Detailed video description text..."}
  ```

- The `videos/metadata.jsonl` file maps video files to their corresponding IDs:
  ```json
  {"file_name": "example.mp4", "id": 0}
  ```

- The `images/metadata.jsonl` file maps image files to their corresponding IDs:
  ```json
  {"file_name": "example.png", "id": 0}
  ```

:::info
- Image and video files are linked by sharing the same ID
- If image files are not provided, the system will default to using the first frame of the corresponding video as the input image
:::

## Notes

- Training sets (`train/`) are used for model training, test sets (`test/`) are used for evaluating model performance

- Each dataset will generate a `.cache/` directory during training, used to store preprocessed data. If the dataset changes, you need to **manually delete this directory** and retrain.
