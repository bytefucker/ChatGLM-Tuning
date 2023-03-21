#! /usr/bin/env bash

set -e

python tokenize_dataset_rows.py \
    --jsonl_path data/alpaca_data.jsonl \
    --save_path data/alpaca \
    --max_seq_length 320