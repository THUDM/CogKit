#!/usr/bin/env bash

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "THUDM/CogView4-6B"
    --model_name "cogview4-6b"  # candidate: ["cogview4-6b"]
    --model_type "t2i"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "/path/to/output"
    --report_to "tensorboard"
)

# Data Configuration
DATA_ARGS=(
    --data_root "/path/to/data"
)

# Training Configuration
TRAIN_ARGS=(
    --seed 42  # random seed
    --train_epochs 1  # number of training epochs

    --learning_rate 2e-5

    # Note: For CogView4 series models, height and width should be **32N** (multiple of 32)
    --train_resolution "1024x1024"  # (height x width)

    #########   Please keep consistent with deepspeed config file ##########
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"]   Note: CogVideoX-2B only supports fp16 training
    ########################################################################

)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory true
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 10  # save checkpoint every x steps
    --checkpointing_limit 2   # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/absolute/path/to/checkpoint_dir"  # if you want to resume from a checkpoint
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_steps 10  # should be multiple of checkpointing_steps
)

# Combine all arguments and launch training
accelerate launch --config_file ../configs/accelerate_config.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"
