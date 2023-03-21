#! /usr/bin/env bash

set -e

nohup python -u finetune.py \
    --dataset_path data/alpaca \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --fp16 \
    --logging_steps 50 \
    --output_dir output &