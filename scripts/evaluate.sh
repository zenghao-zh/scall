#!/bin/bash

# Evaluate CTC-CRF Model
# Usage: bash scripts/evaluate.sh

export CUDA_VISIBLE_DEVICES=4

python opencall_cli/evaluate.py \
    --model_dir /workspace/huada/task_results/index_ddp_lstm_ctc_crf \
    --weight_path /workspace/huada/task_results/index_ddp_lstm_ctc_crf/weights_54.tar \
    --data_dir /workspace/huada/all_refs_label_for_ctc/train_data/train \
    --output_dir /workspace/huada/task_results/evaluation_results \
    --device cuda:0 \
    --val_batch_size 256 \
    --val_size 20000 \
    --tokenization kmer \
    --use_half \
    --seed 25

