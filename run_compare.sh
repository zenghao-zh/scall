#!/bin/bash
set -e

# ---------- 配置 ----------
READS="/workspace/huada/moffett_data/ccf_eval"
REF="/workspace/huada/moffett_data/HG002.fasta"
DEVICE="cuda:0"
OUTPUT_DIR="/workspace/huada/scall/results"
THREADS=4

EXAMPLE="/workspace/huada/scall/cyclonebasecall-moffett_dense/cyclonebasecall/test/example.py"
MINIMAP2="/workspace/huada/scall/eval/cycloneeval/minimap2linux/minimap2"
CALC_ACC="/workspace/huada/scall/eval/cycloneeval/minimap2linux/calculate_accuracy.py"

mkdir -p ${OUTPUT_DIR}

# ---------- Model 1: l9_6x_0214 / weights_best ----------
NAME="io_quant_8"
echo ">>> [1] ${NAME}"

CUDA_VISIBLE_DEVICES=5 python ${EXAMPLE} \
    --model_dir /workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214 \
    --fastq ${OUTPUT_DIR}/${NAME}.fastq \
    --reads ${READS} --device ${DEVICE} \
    --decode viterbi \
    --quant --io_quant /workspace/huada/task_results/lstm_ctc_crf_qat_int8/io_quant_8.pth \
    --act_scales /workspace/huada/task_results/lstm_ctc_crf_qat_int8/act_scales_8.pth

${MINIMAP2} -ax map-ont --eqx -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 \
    -s 180 -U70,1000000 -t ${THREADS} \
    ${REF} ${OUTPUT_DIR}/${NAME}.fastq -o ${OUTPUT_DIR}/${NAME}.sam --secondary=no

python ${CALC_ACC} --sam ${OUTPUT_DIR}/${NAME}.sam --output_png ${OUTPUT_DIR}/${NAME}.png

# ---------- Model 2: ddp_layer8 / weights_49 ----------
# NAME="l5_1x_0214_baseline"
# echo ">>> [2] ${NAME}"

# CUDA_VISIBLE_DEVICES=5 python ${EXAMPLE} \
#     --model_dir /workspace/huada/task_results/lstm_ctc_crf_baseline_0214 \
#     --weights weights_40.tar \
#     --fastq ${OUTPUT_DIR}/${NAME}.fastq \
#     --reads ${READS} --device ${DEVICE}

# ${MINIMAP2} -ax map-ont --eqx -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 \
#     -s 180 -U70,1000000 -t ${THREADS} \
#     ${REF} ${OUTPUT_DIR}/${NAME}.fastq -o ${OUTPUT_DIR}/${NAME}.sam --secondary=no

# python ${CALC_ACC} --sam ${OUTPUT_DIR}/${NAME}.sam --output_png ${OUTPUT_DIR}/${NAME}.png

# ---------- Done ----------
echo ">>> Results:"
ls -lh ${OUTPUT_DIR}/
