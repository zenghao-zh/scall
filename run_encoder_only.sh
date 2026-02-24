cd opencall_cli
bash scripts/train_index_ddp_encoder_only.sh 2 /workspace/huada/hg002_label_for_softmax/train_data/train xxxxx 64 0.0004 10000000000 0 1 lstm_encoder 200 encoder_only2 

bash scripts/train_index_ddp.sh 4 /workspace/huada/moffett_data/250F600274011_train_data/train xxxxx 256 0.0004 10000000000 0 20 lstm_ctc_crf 200 lstm_ctc_crf_kmer

bash scripts/train_index_ddp_sparse.sh 4 /workspace/huada/all_refs_label_for_ctc/train_data/train xxxxx 256 0.0004 10000000000 0 20 lstm_ctc_crf 200 lstm_ctc_crf_kmer_0129_l9-6x-woL

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

python opencall_cli/convert_hdf5_to_pt.py --input_dir /workspace/huada/all_refs_label_for_ctc/train_data/train --output_dir /workspace/huada/train_data_pt/train

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 /workspace/huada/scall/opencall_cli/train_fast.py \
    --data_dir /workspace/huada/train_data_memmap/train \
    --data_type pt \
    --batch_size 256 \
    --lr 0.0004 \
    --epoch_num 20 \
    --model lstm_ctc_crf \
    --output_name lstm_ctc_crf_optimized \
    --num_workers 2 \
    --log_interval 10 \
    --warmup_steps 200 \
    --prune_log ./6x_cgb256_prune


CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4 /workspace/huada/scall/opencall_cli/train_fast.py     --data_dir /workspace/huada/moffett_data/250F600274011_train_data/train_mmap     --data_type pt     --batch_size 256     --lr 0.0004     --epoch_num 20     --model lstm_ctc_crf     --output_name lstm_ctc_crf_optimized_l9_6x_0214     --num_workers 2     --log_interval 10 --val_before_train --resume