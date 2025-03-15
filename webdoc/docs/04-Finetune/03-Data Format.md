---
---

# Dataset Format

<!-- TODO: add link to data dir -->
`src/cogkit/finetune/data` directory contains various dataset templates for fine-tuning different models, please refer to the corresponding dataset template based on your task type:

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

- Each directory contains video files (`.mp4`) and **optional** corresponding image files (`.png`)
- The `metadata.jsonl` file contains metadata information for each sample

    ```json
    {"file_name": "example.mp4", "id": 0, "prompt": "Detailed video description text..."}
    {"file_name": "example.png", "id": 0}  // optional
    ```

    :::info
    - Image files are optional; if not provided, the system will default to using the first frame of the video as the input image
    - When image files are provided, they are associated with the video file of the same name through the id field
    :::

## Notes

- Training sets (`train/`) are used for model training
- Test sets (`test/`) are used for evaluating model performance
- Each dataset will generate a `.cache/` directory during training, used to store preprocessed cache data. If the dataset changes, you need to **manually delete this directory** and retrain.
