#! /usr/bin/env bash

torchrun \
    --nproc_per_node=[number of GPUs] \
    --master_port=29501 \
    ../train.py \
    --yaml config.yaml
