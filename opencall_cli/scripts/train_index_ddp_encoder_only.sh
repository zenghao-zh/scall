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


CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node $GPU_NUMS train_index_ddp_encoder_only.py  \
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

# bash scripts/train_index_ddp_encoder_only.sh 4 /mnt/seqdata/data/pc449-hd110_hg002_labelling_v3_train_data/train_data/train xxxxx 64 0.0004 10000000000 0 1 lstm_encoder 200 encoder_only