cd opencall_cli
bash scripts/train_index_ddp_encoder_only.sh 2 /workspace/huada/hg002_label_for_softmax/train_data/train xxxxx 64 0.0004 10000000000 0 1 lstm_encoder 200 encoder_only2 

bash scripts/train_index_ddp.sh 4 /workspace/huada/all_refs_label_for_ctc/train_data/train xxxxx 256 0.0004 10000000000 0 20 lstm_ctc_crf 200 lstm_ctc_crf_kmer

bash scripts/train_index_ddp_sparse.sh 8 /workspace/huada/all_refs_label_for_ctc/train_data/train xxxxx 256 0.0004 10000000000 0 10 lstm_ctc_crf 200 lstm_ctc_crf_kmer

bash scripts/train_index_ddp_sparse_distill.sh \
    4 \
    /workspace/huada/all_refs_label_for_ctc/train_data/train \
    /workspace/huada/task_results/lstm_ctc_crf_kmer_8x_0105/weights_73.tar \
    256 \
    0.0002 \
    10000000000 \
    0 \
    10 \
    lstm_ctc_crf \
    200 \
    lstm_ctc_crf_kmer_6x_distill_0120 \
    /workspace/huada/task_results/index_ddp_lstm_ctc_crf_layer_8/weights_49.tar \
    3.0 \
    0.1 \
    kl