#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 配置 ----------
READS="/workspace/huada/moffett_data/ccf_eval"
REF="/workspace/huada/moffett_data/HG002.fasta"
MODEL_DIR="/workspace/huada/task_results/lstm_ctc_crf_finetune_moffett_fast"
DEVICE="cuda:0"
GPU_ID=5
OUTPUT_DIR="${SCRIPT_DIR}/results"
THREADS=4
NAME="0318_moffett_finetune_viterbi"

mkdir -p ${OUTPUT_DIR}

# ---------- 1. Basecall ----------
echo ">>> [basecall] ${NAME}"
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_DIR}/example.py \
    --model_dir ${MODEL_DIR} \
    --fastq ${OUTPUT_DIR}/${NAME}.fastq \
    --reads ${READS} --device ${DEVICE} \
    --decode viterbi \
    --quant --bf16 \
    --io_quant ${MODEL_DIR}/io_quant_0318.pth \
    --act_scales ${MODEL_DIR}/act_scales_0318.pth

# ---------- 2. Align ----------
echo ">>> [minimap2] ${NAME}"
${SCRIPT_DIR}/minimap2 -ax map-ont --eqx -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 \
    -s 180 -U70,1000000 -t ${THREADS} \
    ${REF} ${OUTPUT_DIR}/${NAME}.fastq -o ${OUTPUT_DIR}/${NAME}.sam --secondary=no

# ---------- 3. Accuracy ----------
echo ">>> [accuracy] ${NAME}"
python ${SCRIPT_DIR}/calculate_accuracy.py \
    --sam ${OUTPUT_DIR}/${NAME}.sam --output_png ${OUTPUT_DIR}/${NAME}.png

echo ">>> Done. Results:"
ls -lh ${OUTPUT_DIR}/
