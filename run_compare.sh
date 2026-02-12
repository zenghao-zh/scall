#!/bin/bash
set -e

# ---------- 配置 ----------
READS="/workspace/huada/scall/1.fast5"
REF="/workspace/huada/scall/ecoli.fasta"
DEVICE="cuda:0"
OUTPUT_DIR="/workspace/huada/scall/results"
THREADS=4

EXAMPLE="/workspace/huada/scall/cyclonebasecall-moffett_dense/cyclonebasecall/test/example.py"
MINIMAP2="/workspace/huada/scall/eval/cycloneeval/minimap2linux/minimap2"
CALC_ACC="/workspace/huada/scall/eval/cycloneeval/minimap2linux/calculate_accuracy.py"

mkdir -p ${OUTPUT_DIR}

# ---------- Model 1: kmer_0123_67 / weights_59 ----------
NAME="kmer_0123_67_w59"
echo ">>> [1] ${NAME}"

python ${EXAMPLE} \
    --model_dir /workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67 \
    --weights weights_59.tar \
    --fastq ${OUTPUT_DIR}/${NAME}.fastq \
    --reads ${READS} --device ${DEVICE}

${MINIMAP2} -ax map-ont --eqx -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 \
    -s 180 -U70,1000000 -t ${THREADS} \
    ${REF} ${OUTPUT_DIR}/${NAME}.fastq -o ${OUTPUT_DIR}/${NAME}.sam --secondary=no

python ${CALC_ACC} --sam ${OUTPUT_DIR}/${NAME}.sam --output_png ${OUTPUT_DIR}/${NAME}.png

# ---------- Model 2: ddp_layer8 / weights_49 ----------
NAME="ddp_layer8_w49"
echo ">>> [2] ${NAME}"

python ${EXAMPLE} \
    --model_dir /workspace/huada/task_results/index_ddp_lstm_ctc_crf_layer_8 \
    --weights weights_49.tar \
    --fastq ${OUTPUT_DIR}/${NAME}.fastq \
    --reads ${READS} --device ${DEVICE}

${MINIMAP2} -ax map-ont --eqx -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 \
    -s 180 -U70,1000000 -t ${THREADS} \
    ${REF} ${OUTPUT_DIR}/${NAME}.fastq -o ${OUTPUT_DIR}/${NAME}.sam --secondary=no

python ${CALC_ACC} --sam ${OUTPUT_DIR}/${NAME}.sam --output_png ${OUTPUT_DIR}/${NAME}.png

# ---------- Done ----------
echo ">>> Results:"
ls -lh ${OUTPUT_DIR}/
