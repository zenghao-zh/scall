# Scall
用于剔除ctc-crf后，进行不同的解码实验。

## requirements
deploy/docker/requirements.txt

## encoder only(softmax)
1. 单卡
```
python3.8 train_index_lstm_encoder_only.py  # 需保证脚本中--res_prefix文件夹存在
```

2. 多卡
```
cd opencall_cli
# 激活python3.8后，且需保证脚本中--res_prefix文件夹存在
bash scripts/train_index_ddp_encoder_only.sh 4 ../train_data/data_for_encoder_only/train xxxxx 64 0.0004 10000000000 0 1 lstm_encoder 200 encoder_only2
```

## normal(ctc)
1. 单卡
```
python3.8 train_index.py  # 需保证脚本中--res_prefix文件夹存在
```

2. 多卡
```
cd opencall_cli
# 激活python3.8后，且需保证脚本中--res_prefix文件夹存在
bash scripts/train_index_ddp.sh 2 ../train_data/normal_data/train xxxxx 42 0.0004 10000000000 0 1 quartznet 200 quartznet
```