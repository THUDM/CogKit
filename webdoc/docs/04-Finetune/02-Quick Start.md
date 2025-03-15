# Quick Start

## Setup

Please refer to the [installation guide](../02-Installation.md) to setup your environment

<!-- TODO: clone the repo to finetune? clone -->

## Data

Before fine-tuning, you need to prepare your dataset according to the expected format. See the [data format](./03-Data%20Format.md) documentation for details on how to structure your data

## Training

:::info
We recommend that you read the corresponding [model card](../05-Model%20Card.mdx) before starting training to follow the parameter settings requirements and fine-tuning best practices
:::

<!-- TODO: move training script to cli folder? -->
<!-- TODO: add link to corresponding folder -->
1. Navigate to the `src/cogkit/finetune/diffusion` directory

<!-- TODO: add link to training script folder -->
<!-- TODO: add link to train_ddp_t2i.sh -->
2. Choose the appropriate training script from the `scripts` directory based on your task type and distribution strategy. For example, `train_ddp_t2i.sh` corresponds to DDP strategy + text-to-image task

3. Review and adjust the parameters in the selected training script (e.g., `--data_root`, `--output_dir`, etc.)

<!-- TODO: add link to accelerate config -->
4. If you are using ZeRO strategy, refer to `accelerate_config.yaml` to confirm your ZeRO level and number of GPUs

5. Run the script, for example:
   ```bash
   bash scripts/train_ddp_t2i.sh
   ```


## Load Fine-tuned Model

### LoRA

After fine-tuning with LoRA, you can load your trained weights during inference using the `--lora_model_id_or_path` option or parameter. For more details, please refer to the inference guide.

### ZeRO

After fine-tuning with ZeRO strategy, you need to use the `zero_to_fp32.py` script provided in the `scripts` directory to convert the ZeRO checkpoint weights into Diffusers format. For example:

<!-- FIXME: path to zero2diffusers.py? -->
```bash
python zero2diffusers.py checkpoint_dir/ output_dir/ --bfloat16
```

During inference, pass the `output_dir/` to the `--transformer_path` option or parameter. For more details, please refer to the inference guide.
