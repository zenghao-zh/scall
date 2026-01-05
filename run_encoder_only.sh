cd opencall_cli
bash scripts/train_index_ddp_encoder_only.sh 2 /workspace/huada/hg002_label_for_softmax/train_data/train xxxxx 64 0.0004 10000000000 0 1 lstm_encoder 200 encoder_only2 

bash scripts/train_index_ddp.sh 4 /workspace/huada/all_refs_label_for_ctc/train_data/train xxxxx 256 0.0004 10000000000 0 20 lstm_ctc_crf 200 lstm_ctc_crf_kmer