#!/bin/bash

# Number of GPUs
NUM_GPUS=3

# Launch distributed training
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    scripts/train.py \
    --config configs/default.yml