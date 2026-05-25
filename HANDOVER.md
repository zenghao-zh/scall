# 华大 Basecall 项目交接文档

本文档简要说明华大 basecaller 训练 / 量化 / Reads 评估的完整流程，便于后续接手同事快速复现。

## 目录

- [1. 项目概览](#1-项目概览)
- [2. 环境与目录约定](#2-环境与目录约定)
- [3. 训练流程](#3-训练流程)
  - [3.1 Baseline 训练](#31-baseline-训练)
  - [3.2 替换为墨芯适配激活函数后微调](#32-替换为墨芯适配激活函数后微调)
- [4. Chunk 级评估（训练过程中使用）](#4-chunk-级评估训练过程中使用)
- [5. 模型量化到 INT8](#5-模型量化到-int8)
- [6. Reads 级评估（生成 FastQ + 计算 reads 精度）](#6-reads-级评估生成-fastq--计算-reads-精度)
- [7. 验收指标](#7-验收指标)
- [8. 关键脚本 / 文件速查](#8-关键脚本--文件速查)

---

## 1. 项目概览

整体流程：

```
Baseline 训练（普通 LSTM）
        │
        ▼
替换墨芯适配激活函数 → 再微调 1 个 epoch
        │
        ▼
量化为 INT8（lstmIOquant_jjy.py）
        │
        ▼
Reads 级评估（reads_evaluation/run_evaluation.sh）
        │
        ▼
生成 fastq + reads 精度报告（要求 < 0.1%）
```

- 训练入口：`opencall_cli/train_fast.py`
- 量化脚本：`pruning/lstmIOquant_jjy.py`
- Reads 评估框架：`reads_evaluation/`
- 训练 / 微调脚本入口：`run_train.sh`
- Reads 评估脚本入口：`reads_evaluation/run_evaluation.sh`

## 2. 环境与目录约定

- 代码根目录：`/workspace/huada/scall`
- 训练数据目录：`/workspace/huada/moffett_data/250F600274011_train_data/train_mmap`
- 任务输出根目录：`/workspace/huada/task_results/<output_name>/`
  - 训练权重：`weights_<epoch>.tar`
  - 配置：`config.toml`
  - 量化产物：`io_quant_0318.pth`、`act_scales_0318.pth`
- Reads 评估数据：
  - Reads 输入：`/workspace/huada/moffett_data/ccf_eval`
  - 参考基因组：`/workspace/huada/moffett_data/HG002.fasta`

## 3. 训练流程

完整命令见 `run_train.sh`，分两步：

### 3.1 Baseline 训练

使用普通 LSTM 训练 baseline，8 卡 DDP：

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" \
torchrun --nproc_per_node=8 /workspace/huada/scall/opencall_cli/train_fast.py \
    --data_dir /workspace/huada/moffett_data/250F600274011_train_data/train_mmap \
    --data_type pt \
    --batch_size 128 \
    --lr 0.0004 \
    --epoch_num 20 \
    --model lstm_ctc_crf \
    --output_name lstm_ctc_crf_optimized_l9_6x_0214 \
    --num_workers 2 \
    --log_interval 10 \
    --val_before_train
```

产物保存在 `/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/`，挑选效果最好的 epoch（例如 `weights_40.tar`）作为下一步的预训练权重。

### 3.2 替换为墨芯适配激活函数后微调

> **重要**：baseline 训练好之后，需要把 LSTM 内部激活函数替换为「墨芯适配精度的激活函数」（通过 `--use_fast_lstm` 开启），再用较小学习率微调 1 个 epoch 即可（脚本里写的 10 个 epoch 是上限，实际跑 1 个 epoch 就够，按 val 精度选最好的）。

```bash
CUDA_VISIBLE_DEVICES="0,1,2,3" \
torchrun --nproc_per_node=4 /workspace/huada/scall/opencall_cli/train_fast.py \
    --data_dir /workspace/huada/moffett_data/250F600274011_train_data/train_mmap \
    --data_type pt \
    --batch_size 128 \
    --lr 0.0001 \
    --epoch_num 10 \
    --model lstm_ctc_crf \
    --output_name lstm_ctc_crf_finetune_moffett_fast \
    --num_workers 2 \
    --log_interval 10 \
    --pre_trained_params_file /workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/weights_40.tar \
    --print_model \
    --use_fast_lstm \
    --val_before_train
```

关键参数说明：

| 参数 | 含义 |
| --- | --- |
| `--pre_trained_params_file` | baseline 模型权重路径 |
| `--use_fast_lstm` | 替换为墨芯适配精度激活函数的 LSTM（必须开） |
| `--val_before_train` | 微调前先评估一次，便于看到替换激活函数后的初始精度跌幅 |
| `--lr 0.0001` | 微调用小学习率 |

产物：`/workspace/huada/task_results/lstm_ctc_crf_finetune_moffett_fast/weights_*.tar`

## 4. Chunk 级评估（训练过程中使用）

训练过程中 / 训完之后做 chunk 级别 val 精度的脚本：`run_evaluation.sh`，内部调用 `opencall_cli/evaluate.py`。

```bash
bash /workspace/huada/scall/run_evaluation.sh
```

脚本内部命令（按需修改路径）：

```bash
MODEL_DIR="/workspace/huada/task_results/index_ddp_lstm_ctc_crf"
WEIGHT_PATH="/workspace/huada/task_results/index_ddp_lstm_ctc_crf/weights_5.tar"
DATA_DIR="/workspace/basecall_data/train_data/index/test_index/train"
OUTPUT_DIR="/workspace/huada/task_results/evaluation_results"
DEVICE="cuda:0"

python /workspace/huada/scall/opencall_cli/evaluate.py \
    --model_dir ${MODEL_DIR} \
    --weight_path ${WEIGHT_PATH} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --device ${DEVICE} \
    --val_batch_size 16 \
    --val_size 20000 \
    --tokenization kmer \
    --use_half \
    --seed 25
```

使用时只需替换 `MODEL_DIR` / `WEIGHT_PATH` / `DATA_DIR` 三处即可。该评估和 `train_fast.py` 训练中的 val 流程一致，输出 chunk 上的 mean / median 精度。

## 5. 模型量化到 INT8

> 训好（含 `--use_fast_lstm` 微调）的模型需要用 `pruning/lstmIOquant_jjy.py` 校准并量化到 INT8。

运行：

```bash
cd /workspace/huada/scall
python pruning/lstmIOquant_jjy.py
```

脚本内部要在 `main()` 顶部按需修改路径：

```python
config_file = '.../lstm_ctc_crf_finetune_moffett_fast/config.toml'
pretrained_model_file = '.../lstm_ctc_crf_finetune_moffett_fast/weights_0.tar'
act_scales_path = '.../lstm_ctc_crf_finetune_moffett_fast/act_scales_0318.pth'
io_quant_path = '.../lstm_ctc_crf_finetune_moffett_fast/io_quant_0318.pth'
new_io_quant_path = '.../lstm_ctc_crf_finetune_moffett_fast/io_quant_wo0_0318.pth'
```

该脚本做了 4 件事：

1. 加载训好的 fast-LSTM 模型；
2. 用 forward hook 在所有 `LSTM` / `Linear` 层统计输入输出的绝对值最大值，得到每层激活的 scale（`act_scales`）；
3. 在每个 LSTM 后插入 `FakeQuant`（8-bit）层，构成量化模型；
4. 同时跑量化模型和原始模型在同一 val 集上的 chunk 精度做对比，并保存：
   - `act_scales_0318.pth`：每层 input / output 的最大值
   - `io_quant_0318.pth`：插入 FakeQuant 后的完整 state_dict
   - `io_quant_wo0_0318.pth`：去掉冗余 `.0` / `lstm.` 命名层级、用于部署的 state_dict

如果想直接复用量化模型做推理，可以参考 `pruning/load_io_quant.py` 里的 `load_io_quant_model()` 实现：先 build 网络 → 加载 act_scales → 插 FakeQuant → load `io_quant.pth`。

## 6. Reads 级评估（生成 FastQ + 计算 reads 精度）

`reads_evaluation/` 是华大要求的 reads 级评估框架。脚本入口：

```bash
bash /workspace/huada/scall/reads_evaluation/run_evaluation.sh
```

`run_evaluation.sh` 完整流程：

```bash
READS="/workspace/huada/moffett_data/ccf_eval"
REF="/workspace/huada/moffett_data/HG002.fasta"
MODEL_DIR="/workspace/huada/task_results/lstm_ctc_crf_finetune_moffett_fast"
DEVICE="cuda:0"
GPU_ID=5
OUTPUT_DIR="${SCRIPT_DIR}/results"
THREADS=4
NAME="0521_moffett_finetune_viterbi_bf16"

# 1. Basecall：用量化模型在 fast5 上推理，输出 fastq
CUDA_VISIBLE_DEVICES=${GPU_ID} python ${SCRIPT_DIR}/example.py \
    --model_dir ${MODEL_DIR} \
    --fastq ${OUTPUT_DIR}/${NAME}.fastq \
    --reads ${READS} --device ${DEVICE} \
    --decode viterbi \
    --quant --bf16 \
    --io_quant ${MODEL_DIR}/io_quant_0318.pth \
    --act_scales ${MODEL_DIR}/act_scales_0318.pth

# 2. Align：minimap2 比对到参考基因组
${SCRIPT_DIR}/minimap2 -ax map-ont --eqx -k 16 -w 13 -A 2 -B 4 -O 4,41 -E 2,1 \
    -s 180 -U70,1000000 -t ${THREADS} \
    ${REF} ${OUTPUT_DIR}/${NAME}.fastq -o ${OUTPUT_DIR}/${NAME}.sam --secondary=no

# 3. Accuracy：从 sam 文件统计 reads 上的精度并出图
python ${SCRIPT_DIR}/calculate_accuracy.py \
    --sam ${OUTPUT_DIR}/${NAME}.sam --output_png ${OUTPUT_DIR}/${NAME}.png
```

常用参数说明（`example.py`）：

| 参数 | 含义 |
| --- | --- |
| `--model_dir` | 量化后模型所在目录（需含 `config.toml`、`weights.tar`、`io_quant_*.pth`、`act_scales_*.pth`） |
| `--reads` | fast5 输入目录 |
| `--fastq` | 输出 fastq 路径 |
| `--decode` | `viterbi` 或 `beam_search`（koi） |
| `--quant` | 启用 FakeQuant int8 量化推理 |
| `--bf16` | 用 bf16 跑（替换为 `ManualLSTMRNN`） |
| `--io_quant` / `--act_scales` | 量化产物路径，默认会在 `model_dir` 下自动检测 |

输出：
- `results/<NAME>.fastq`：basecall 结果
- `results/<NAME>.sam`：比对结果
- `results/<NAME>.png`：reads 精度分布图（这是交付给华大的关键产物）

切换模型时只需要改 `MODEL_DIR` 和 `NAME` 即可，io_quant / act_scales 默认从 model_dir 下读取。

## 7. 验收指标

> **华大要求：reads 上的错误率 < 0.1%**（即 reads 精度 > 99.9%）。

精度看 `reads_evaluation/calculate_accuracy.py` 生成的 PNG 中的 median / mean。

## 8. 关键脚本 / 文件速查

| 路径 | 作用 |
| --- | --- |
| `run_train.sh` | 训练 + 微调的 shell 命令模板 |
| `opencall_cli/train_fast.py` | DDP 训练主入口；含 `replace_lstm_with_fast()`（`--use_fast_lstm` 时调用） |
| `run_evaluation.sh` | chunk 级 val 精度评估（训练过程同样的流程） |
| `opencall_cli/evaluate.py` | chunk 级评估底层实现 |
| `pruning/lstmIOquant_jjy.py` | INT8 量化校准脚本（产出 `io_quant_*.pth`、`act_scales_*.pth`） |
| `pruning/load_io_quant.py` | 加载量化模型的参考实现 |
| `reads_evaluation/run_evaluation.sh` | Reads 级评估完整 pipeline（basecall + minimap2 + 精度统计） |
| `reads_evaluation/example.py` | Reads 级 basecall 入口（支持 `--quant` / `--bf16` / 解码方式切换） |
| `reads_evaluation/calculate_accuracy.py` | 从 SAM 算 reads 精度并出图 |
| `reads_evaluation/minimap2` | 已编译好的 minimap2 二进制 |

---

### 常用一键复现命令

```bash
# 1. baseline 训练
bash /workspace/huada/scall/run_train.sh  # 注意手动只跑 baseline 那一行

# 2. 替换激活函数 + 微调 1 epoch
#    （同样在 run_train.sh 里，跑带 --use_fast_lstm 的那一行）

# 3. chunk 级评估（按需改 run_evaluation.sh 内 MODEL_DIR / WEIGHT_PATH / DATA_DIR）
bash /workspace/huada/scall/run_evaluation.sh

# 4. INT8 量化（按需改脚本顶部路径）
python /workspace/huada/scall/pruning/lstmIOquant_jjy.py

# 5. Reads 级评估（按需改 run_evaluation.sh 内 MODEL_DIR / NAME）
bash /workspace/huada/scall/reads_evaluation/run_evaluation.sh
```
