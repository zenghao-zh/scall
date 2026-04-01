## baseline训练
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun --nproc_per_node=8 /workspace/huada/scall/opencall_cli/train_fast.py     --data_dir 
/workspace/huada/moffett_data/250F600274011_train_data/train_mmap     --data_type pt     --batch_
size 128     --lr 0.0004     --epoch_num 20     --model lstm_ctc_crf     --output_name lstm_ctc_c
rf_optimized_l9_6x_0214     --num_workers 2     --log_interval 10 --val_before_train 

## 微调moffett_fast
CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun --nproc_per_node=4   /workspace/huada/scall/opencall_cli/train_fast.py   --data_dir /workspace/huada/moffett_data/250F600274011_train_data/train_mmap   --data_type pt   --batch_size 128   --lr 0.0001   --epoch_num 10   --model lstm_ctc_crf   --output_name lstm_ctc_crf_finetune_moffett_fast   --num_workers 2   --log_interval 10   --pre_trained_params_file /workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/weights_40.tar   --print_model --use_fast_lstm --val_before_train