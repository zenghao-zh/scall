#!/bin/bash

# Model Evaluation Script
# This script runs model evaluation on validation data

# Example usage:
# ./run_evaluation.sh

# Configuration
MODEL_DIR="/workspace/huada/task_results/index_ddp_lstm_ctc_crf"
WEIGHT_PATH="/workspace/huada/task_results/index_ddp_lstm_ctc_crf/weights_5.tar"
DATA_DIR="/workspace/basecall_data/train_data/index/test_index/train"
OUTPUT_DIR="/workspace/huada/task_results/evaluation_results"
DEVICE="cuda:0"

# Run evaluation
python /workspace/huada/scall/opencall_cli/evaluate.py \
    --model_dir ${MODEL_DIR} \
    --weight_path ${WEIGHT_PATH} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --val_batch_size 16 \
    --val_size 20000 \
    --tokenization kmer \
    --use_half \
    --seed 25

echo "Evaluation completed!"

