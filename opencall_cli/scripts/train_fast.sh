#!/bin/bash
set -x

# ============================================================================
# 高效训练脚本
# 
# 特性:
#   - 梯度累积支持
#   - CUDA 数据预取
#   - torch.compile 加速
#   - 断点续训
#   - 进度条显示
# ============================================================================

# Get script directory for relative imports
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# ============================================================================
# 参数设置
# ============================================================================

# 必需参数
GPU_NUMS=${1:-4}
DATA_DIR=${2:-"/workspace/huada/all_refs_label_for_ctc/train_data/train"}
OUTPUT_NAME=${3:-"exp_$(date +%Y%m%d_%H%M%S)"}

# 训练参数
BATCH_SIZE=${BATCH_SIZE:-256}
GRAD_ACCUM=${GRAD_ACCUM:-1}           # 梯度累积步数
LR=${LR:-0.0004}
EPOCH_NUM=${EPOCH_NUM:-20}
WARMUP_STEPS=${WARMUP_STEPS:-200}
MODEL=${MODEL:-"lstm_ctc_crf"}

# 数据参数
DATA_TYPE=${DATA_TYPE:-"index"}       # "index" (HDF5) 或 "pt" (预转换)
NUM_WORKERS=${NUM_WORKERS:-4}

# 剪枝参数
PRUNING=${PRUNING:-1}
SPARSITY=${SPARSITY:-0.833333}

# 可选参数
PRE_TRAINED=${PRE_TRAINED:-""}
USE_COMPILE=${USE_COMPILE:-0}         # 是否使用 torch.compile
RESUME=${RESUME:-0}                    # 是否断点续训
LOG_INTERVAL=${LOG_INTERVAL:-100}

# ============================================================================
# 构建命令
# ============================================================================

# Find an available port for distributed training
find_free_port() {
    python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
}
MASTER_PORT=${MASTER_PORT:-$(find_free_port)}

# 构建参数列表
CMD="CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=$GPU_NUMS --master_port=$MASTER_PORT train_fast.py"
CMD="$CMD --data_dir $DATA_DIR"
CMD="$CMD --output_name $OUTPUT_NAME"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --grad_accum $GRAD_ACCUM"
CMD="$CMD --lr $LR"
CMD="$CMD --epoch_num $EPOCH_NUM"
CMD="$CMD --warmup_steps $WARMUP_STEPS"
CMD="$CMD --model $MODEL"
CMD="$CMD --data_type $DATA_TYPE"
CMD="$CMD --num_workers $NUM_WORKERS"
CMD="$CMD --pruning $PRUNING"
CMD="$CMD --sparsity $SPARSITY"
CMD="$CMD --log_interval $LOG_INTERVAL"
CMD="$CMD --use_amp"
# 默认不使用 prefetch 以保持数值一致性，如需开启可设置 USE_PREFETCH=1
if [ "${USE_PREFETCH:-0}" -eq 1 ]; then
    CMD="$CMD --use_prefetch"
fi

# 可选参数
if [ -n "$PRE_TRAINED" ] && [ "$PRE_TRAINED" != "xxxxx" ]; then
    CMD="$CMD --pre_trained_params_file $PRE_TRAINED"
fi

if [ "$USE_COMPILE" -eq 1 ]; then
    CMD="$CMD --compile"
fi

if [ "$RESUME" -eq 1 ]; then
    CMD="$CMD --resume"
fi

# ============================================================================
# 执行训练
# ============================================================================

echo "============================================"
echo "Training Configuration:"
echo "  GPU nums:      $GPU_NUMS"
echo "  Data dir:      $DATA_DIR"
echo "  Data type:     $DATA_TYPE"
echo "  Output name:   $OUTPUT_NAME"
echo "  Batch size:    $BATCH_SIZE x $GRAD_ACCUM (effective: $((BATCH_SIZE * GRAD_ACCUM * GPU_NUMS)))"
echo "  Learning rate: $LR"
echo "  Epochs:        $EPOCH_NUM"
echo "  Model:         $MODEL"
echo "  Pruning:       $PRUNING (sparsity: $SPARSITY)"
echo "============================================"

eval $CMD


# ============================================================================
# 使用示例
# ============================================================================

# 基础训练:
#   bash scripts/train_fast.sh 4 /workspace/huada/train_data/train my_exp

# 使用预转换数据 (更快):
#   DATA_TYPE=pt bash scripts/train_fast.sh 4 /workspace/huada/train_data_pt/train my_exp

# 使用梯度累积 (等效于更大的 batch size):
#   GRAD_ACCUM=2 bash scripts/train_fast.sh 4 /path/to/data my_exp

# 使用 torch.compile 加速:
#   USE_COMPILE=1 bash scripts/train_fast.sh 4 /path/to/data my_exp

# 断点续训:
#   RESUME=1 bash scripts/train_fast.sh 4 /path/to/data my_exp

# 完整示例:
#   DATA_TYPE=pt BATCH_SIZE=256 GRAD_ACCUM=2 EPOCH_NUM=20 \
#   bash scripts/train_fast.sh 4 /workspace/huada/train_data_pt/train lstm_exp_v1
