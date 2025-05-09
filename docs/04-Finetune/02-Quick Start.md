# Quick Start

## Setup

* Please refer to the [installation guide](../02-Installation.md) to setup your environment first

* Install finetune dependencies:

   ```bash
   pip install "cogkit[finetune]@git+https://github.com/THUDM/CogKit.git"
   ```

* We provide various training scripts and example datasets in the `CogKit/quickstart` directory. Please clone the repository before training:

   ```bash
   git clone https://github.com/THUDM/CogKit.git
   ```

## Data

Before fine-tuning, you need to prepare your dataset according to the expected format. See the [data format](./03-Data%20Format.md) documentation for details on how to structure your data

## Training

:::info
We recommend that you read the corresponding [model card](../05-Model%20Card.mdx) before starting training to follow the parameter settings requirements and fine-tuning best practices
:::

1. Navigate to the `CogKit/` directory after cloning the repository
   ```bash
   cd CogKit/
   ```

2. Choose the appropriate sub directory from the `quickstart/scripts` based on your task type and distribution strategy. For example, `t2i` corresponds to text-to-image task

3. Review and adjust the parameters in `config.yaml` in the selected training directory

4. Run the script in selected directory:

   ```bash
   bash start_train.sh
   ```

## Load Fine-tuned Model

### Merge Checkpoint

After fine-tuning, you need to use the `merge.py` script provided in the `quickstart/tools/converters` directory to merge the distributed checkpoint weights into a single checkpoint (**except for QLoRA fine-tuning**). For example:

```bash
cd quickstart/tools/converters
python merge.py --checkpoint_dir ckpt/ --output_dir output_dir/
# Add --lora option if you are using LoRA fine-tuning
```

### Load Checkpoint

You can pass the `output_dir` to the `--lora_model_id_or_path` option if you are using LoRA fine-tuning, or to the `--transformer_path` option if you are using FSDP fine-tuning. For more details, please refer to the inference guide.
