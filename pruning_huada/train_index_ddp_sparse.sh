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

CUDA_VISIBLE_DEVICES="6,7" /opt/conda/bin/python -m torch.distributed.launch --nproc_per_node $GPU_NUMS train_index_ddp_sparse.py  \
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


#  bash scripts/train_index_ddp_sparse.sh 4 /store/zjj/tmp/20241231185005_pc28-hd65-wy-1_train_1/train /store/data/old_bgi_models/pytorch/wy_basic_v0.3/wy_basic_v0.3/weights_0.tar 64 0.0004 10000000000 0 3 lstm_ctc_crf 200 wy_basic_v0.3_12x_sparse
#  bash scripts/train_index_ddp_sparse.sh 2 /store/zjj/tmp/20241231185005_pc28-hd65-wy-1_train_1/train /store/zjj/task_results/wy_basic_v0.3_12x_sparse_3epoch/weights_0.tar 64 0.0004 10000000000 0 3 lstm_ctc_crf 200 wy_basic_v0.3_12x_sparse
#  bash scripts/train_index_ddp_sparse.sh 2 /store/zjj/tmp/20241231185005_pc28-hd65-wy-1_train_1/train xxxxx 32 0.0004 100000 0 1 lstm_ctc_crf 200 050328
