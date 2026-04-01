python analyze_spu_gpu_diff.py \
    --data_dir /workspace/huada/moffett_data/lstm_train_dataset_result \
    --file_id 201 \
    --model lstm_ctc_crf \
    --pre_trained_params_file /workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/weights_40.tar \
    --device cuda:0 \
    --num_samples 5 \
    --batch_size 32


python visualize_spu_gpu_diff.py \
    --data_file /workspace/huada/scall/spu_gpu_backbone_outputs_201.pt \
    --sample_idx 454 \
    --save_dir /workspace/huada/scall/visualizations