#!/bin/bash
set -x


GPU_NUMS=$1
DATA_DIR=$2
PARAMS_FILE=$3
BATCH_SIZE=$4
LR=$5
MAX_BATCH_NUM_FOR_TRAINING=$6
DEBUG=$7
EPOCH_NUM=$8
MODEL=$9
WARMUP_STEPS=${10}
OUTPUT_NAME=${11}


CUDA_VISIBLE_DEVICES="2,3,6,7" python -m torch.distributed.launch --nproc_per_node $GPU_NUMS train_index_ddp.py  \
	--data_dir $DATA_DIR \
	--pre_trained_params_file      $PARAMS_FILE \
	--batch_size         $BATCH_SIZE \
	--lr         $LR \
	--limit_train_size $MAX_BATCH_NUM_FOR_TRAINING  \
	--debug $DEBUG  \
	--epoch_num $EPOCH_NUM  \
	--model $MODEL \
	--seed 25 \
	--warmup_steps $WARMUP_STEPS \
	--output_name $OUTPUT_NAME

# bash scripts/train_index_ddp.sh 2 /mnt/seqdata/zjj_data/datasets/pc449-hd110_all_refs_train_data/small_train_data/train xxxxx 42 0.0004 10000000000 0 1 quartznet 200 quartznet
