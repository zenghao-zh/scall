"""
分析 SPU 和 GPU forward 输出的差异

从 train_crfencoder.py 提取核心逻辑:
1. 数据加载 (SPU backbone 预计算输出 + 原始数据)
2. SPU forward 路径: 直接使用 spu_backbone_output (不经过 encoder)
3. GPU forward 路径: raw_data -> backbone (不经过 encoder)
4. 对比两者的 backbone 输出差异

使用示例:

1. 使用普通预训练模型:
    python analyze_spu_gpu_diff.py \
        --data_dir /workspace/huada/moffett_data/lstm_train_dataset_result \
        --file_id 201 \
        --model lstm_ctc_crf \
        --pre_trained_params_file /workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/weights_40.tar \
        --device cuda:0 \
        --num_samples 5 \
        --batch_size 32

2. 使用量化模型:
    python analyze_spu_gpu_diff.py \
        --data_dir /workspace/huada/moffett_data/lstm_train_dataset_result \
        --file_id 201 \
        --model lstm_ctc_crf \
        --use_quant \
        --act_scales_path /workspace/huada/scall/caoyu/layer_9_6x_act_scales_0305.pth \
        --io_quant_path /workspace/huada/scall/caoyu/layer_9_6x_io_quant_0305.pth \
        --bitwidth 8 \
        --device cuda:0 \
        --num_samples 5 \
        --batch_size 32
"""

import sys
import os
pro_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pro_dir)

import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from collections import OrderedDict
from torch.nn import Module
from torch.nn.init import orthogonal_
from opencall.utils.util import network

from MoffettLSTM.custom_lstm import FastLSTM

# ============================================================================
# RNN Wrapper (从 viterbi_0224.py 提取)
# ============================================================================

class RNNWrapper(Module):
    """Wrapper for RNN layers (LSTM, GRU, etc.)"""
    def __init__(self, rnn_type, size, insize, bias=True, reverse=False, dropout=0.0):
        super().__init__()
        self.rnn = rnn_type(insize, size, bias=bias)
        self.reverse = reverse
        self.dropout = dropout
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, state=None):
        # FastLSTM with moffett activation ONLY supports bf16
        if isinstance(self.rnn, FastLSTM) and self.rnn.activation_impl == "moffett":
            if x.dtype != torch.bfloat16:
                x = x.to(torch.bfloat16)
        if self.reverse:
            x = x.flip([0])
        y, state_out = self.rnn(x, state)
        if self.reverse:
            y = y.flip([0])
        if self.dropout > 0.0:
            y = self.dropout_layer(y)
        return y

    def init_truncated(self, types=True):
        """Truncated normal initialization"""
        if not types:
            return
        if types is True:
            types = ("weight_ih", "weight_hh")
        for name, param in self.rnn.named_parameters():
            if any(k in name for k in types):
                with torch.no_grad():
                    # Simple truncated normal approximation
                    param.normal_(0, 0.5)
                    param.clamp_(-2, 2)

    def init_orthogonal(self, types=True):
        if not types:
            return
        if types is True:
            types = ("weight_ih", "weight_hh")
        for name, x in self.rnn.named_parameters():
            if any(k in name for k in types):
                for i in range(0, x.size(0), self.rnn.hidden_size):
                    orthogonal_(x[i : i + self.rnn.hidden_size])

    def disable_state_bias(self):
        for name, x in self.rnn.named_parameters():
            if "bias_hh" in name:
                x.requires_grad = False
                x.zero_()


class LSTM(RNNWrapper):
    """LSTM wrapper compatible with opencall's architecture"""
    def __init__(self, size, insize, bias=True, reverse=False, dropout=0.0):
        super().__init__(torch.nn.LSTM, size, insize, bias=bias, reverse=reverse, dropout=dropout)


def replace_lstm_with_manual(model):
    """Replace all torch.nn.LSTM inside LSTM (RNNWrapper) with FastLSTM.

    Copies weights so the model produces identical results.
    Must be called *before* converting to bfloat16.
    """
    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, 'rnn') and isinstance(module.rnn, torch.nn.LSTM):
            old = module.rnn
            fast = FastLSTM.from_torch_lstm(old, activation_impl="moffett")
            module.rnn = fast
            replaced += 1
            print(f"  Replaced {name}.rnn: torch.nn.LSTM → FastLSTM")
    print(f"[replace_lstm_with_manual] Total replaced: {replaced} torch.nn.LSTM → FastLSTM")
    return model


# ============================================================================
# 量化支持 (从 viterbi_0224.py 提取)
# ============================================================================

class FakeQuant(nn.Module):
    """Fake quantization layer for activations."""
    def __init__(self, num_bits, max_val):
        super(FakeQuant, self).__init__()
        self.scale = max_val / (2**(num_bits - 1) - 1)
        self.num_bits = num_bits

    def forward(self, x):
        scale = self.scale.to(dtype=x.dtype, device=x.device)
        x = torch.clamp(x / scale, -2**(self.num_bits - 1), 2**(self.num_bits - 1) - 1)
        x = torch.round(x)
        x = x * scale
        return x


def insert_fakequant_encoder(model, act_scales, bitwidth, device):
    """
    Insert FakeQuant layers around encoder's backbone layers (不包括最后的 LinearCRFEncoder).
    
    参考 viterbi_0224.py 的 insert_fakequant_backbone 实现.
    
    在 opencall 模型中:
        - encoder[:-1] 是 backbone 层 (CNN + LSTM)
        - encoder[-1] 是 LinearCRFEncoder
    
    只对 backbone 层添加 FakeQuant.
    """
    encoder = model.encoder
    inserted_count = 0
    
    # 只处理 backbone 层，不包括最后的 LinearCRFEncoder
    num_backbone_layers = len(encoder) - 1
    
    for i in range(num_backbone_layers):
        scale_key = f'encoder.{i}'
        if scale_key in act_scales and "output" in act_scales[scale_key]:
            # 将原来的 layer 包装在 Sequential 中, 后面加上 FakeQuant
            original_layer = encoder[i]
            encoder[i] = nn.Sequential(
                original_layer,
                FakeQuant(bitwidth, act_scales[scale_key]["output"].to(device))
            )
            inserted_count += 1
            print(f"  插入 FakeQuant 到 encoder[{i}] (backbone 层)")
    
    print(f"共插入 {inserted_count} 个 FakeQuant 层到 backbone")
    return model


def match_names(state_dict, model):
    """
    Match weight names between checkpoint and model by shape sorting.
    从 viterbi_0224.py 复制
    """
    keys_and_shapes = lambda sd: zip(
        *[(k, s) for s, i, k in sorted(
            [(v.shape, i, k) for i, (k, v) in enumerate(sd.items())]
        )]
    )
    k1, s1 = keys_and_shapes(state_dict)
    k2, s2 = keys_and_shapes(model.state_dict())
    assert s1 == s2, "Model architecture does not match checkpoint weights!"
    remap = dict(zip(k1, k2))
    return OrderedDict([(k, remap[k]) for k in state_dict.keys()])


# ============================================================================
# 数据加载逻辑 (从 train_crfencoder.py 提取)
# ============================================================================

def _transform_spu(raw):
    """
    SPU backbone dequant 输出变换:
      raw shape (4,30,32,3,128,256)
      -> permute([1,2,0,4,3,5]) -> (30,32,4,128,3,256)
      -> reshape(-1, N, features)  -> (960, 512, 768)
    """
    permuted = raw.permute(1, 2, 0, 4, 3, 5)
    T = permuted.shape[0] * permuted.shape[1]
    N = permuted.shape[2] * permuted.shape[3]
    F = permuted.shape[4] * permuted.shape[5]
    return permuted.reshape(T, N, F)


def load_data_sample(data_dir, file_id):
    """
    加载一个数据样本 (文件级)
    
    Returns:
        spu_backbone_output: (T, N, F) SPU backbone 的预计算输出
        raw_data: (N, ...) 原始输入数据
        targets: (N,) 目标序列
        lengths: (N,) 序列长度
    """
    spu_path = os.path.join(data_dir, f"spu_backbone_dequant_{file_id}.pt")
    target_path = os.path.join(data_dir, f"target_{file_id}.pt")
    lengths_path = os.path.join(data_dir, f"lengths_{file_id}.pt")
    data_path = os.path.join(data_dir, f"data_{file_id}.pt")
    
    print(f"\n{'='*80}")
    print(f"加载数据文件:")
    print(f"  SPU backbone: {spu_path}")
    print(f"  Raw data: {data_path}")
    print(f"  Target: {target_path}")
    print(f"  Lengths: {lengths_path}")
    
    # 加载 SPU backbone 预计算输出
    raw_spu = torch.load(spu_path, map_location="cpu")
    print(f"\n原始 SPU backbone 输出 shape: {raw_spu.shape}")
    spu_output = _transform_spu(raw_spu).float()
    print(f"变换后 SPU backbone 输出 shape: {spu_output.shape}")
    del raw_spu
    
    # 加载原始数据
    if os.path.exists(data_path):
        raw_data = torch.load(data_path, map_location="cpu").float()
        print(f"原始数据 shape: {raw_data.shape}")
    else:
        raw_data = None
        print(f"警告: 原始数据文件不存在")
    
    # 加载标签
    targets = torch.load(target_path, map_location="cpu")
    lengths = torch.load(lengths_path, map_location="cpu")
    print(f"Targets shape: {targets.shape}")
    print(f"Lengths shape: {lengths.shape}")
    
    return spu_output, raw_data, targets, lengths


def list_available_files(data_dir):
    """列出所有可用的数据文件ID"""
    spu_files = sorted(glob.glob(os.path.join(data_dir, "spu_backbone_dequant_*.pt")))
    file_ids = []
    for spu_path in spu_files:
        import re
        m = re.search(r"spu_backbone_dequant_(\d+)\.pt$", spu_path)
        if m:
            fid = m.group(1)
            target_path = os.path.join(data_dir, f"target_{fid}.pt")
            lengths_path = os.path.join(data_dir, f"lengths_{fid}.pt")
            data_path = os.path.join(data_dir, f"data_{fid}.pt")
            if os.path.exists(target_path) and os.path.exists(lengths_path):
                has_raw = os.path.exists(data_path)
                file_ids.append((fid, has_raw))
    return file_ids


# ============================================================================
# Forward 逻辑 (从 train_crfencoder.py 提取)
# ============================================================================

def spu_forward(spu_backbone_output, device):
    """
    SPU forward 路径: 直接使用预计算的 SPU backbone 输出 (float32)
    
    Args:
        spu_backbone_output: (T, N, F) SPU backbone 的预计算输出
        device: 设备
    
    Returns:
        output: (T, N, F) SPU backbone 输出 (float32)
    """
    return spu_backbone_output.float().to(device)


def gpu_forward(model, raw_data, device, batch_size=32):
    """
    GPU forward 路径: 从原始数据开始，只经过 backbone (encoder 的前 n-1 层)
    
    在 opencall 模型中:
        - model.encoder 是 Serial (继承自 Sequential)，包含所有层
        - encoder[:-1] 是 backbone (CNN + LSTM)
        - encoder[-1] 是 LinearCRFEncoder
    
    FastLSTM (activation_impl="moffett") 只支持 bfloat16 输入，
    因此在逐层 forward 时，会在进入 LSTM 层前强制转换为 bf16。
    
    Args:
        model: 完整模型
        raw_data: (N, ...) 原始输入数据
        device: 设备
        batch_size: mini-batch 大小 (防止显存不足)
    
    Returns:
        output: (T, N, F) backbone 输出
    """
    encoder = model.encoder
    
    all_layers = list(encoder._modules.values())
    backbone_layers = all_layers[:-1]
    
    N = raw_data.shape[0]
    total_batches = (N + batch_size - 1) // batch_size
    all_outputs = []
    
    print(f"  FastLSTM C 扩展首次调用时需要 JIT 编译，请耐心等待...")
    
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch_idx = i // batch_size + 1
            t0 = time.time()
            
            x = raw_data[i:i+batch_size].to(device=device, dtype=torch.bfloat16)
            
            for layer in backbone_layers:
                x = layer(x)
            
            all_outputs.append(x.float().cpu())
            elapsed = time.time() - t0
            print(f"  批次 {batch_idx}/{total_batches} 完成 (耗时 {elapsed:.1f}s), "
                  f"输出 shape={all_outputs[-1].shape}, dtype={all_outputs[-1].dtype}")
            del x
        
        if len(all_outputs) > 1:
            full_output = torch.cat(all_outputs, dim=1)
        else:
            full_output = all_outputs[0]
    
    return full_output.to(device)


# ============================================================================
# 差异分析
# ============================================================================

def analyze_difference(spu_scores, gpu_scores, sample_indices=None):
    """
    分析 SPU 和 GPU 输出的差异
    
    Args:
        spu_scores: (T, N, C) SPU forward 的输出
        gpu_scores: (T, N, C) GPU forward 的输出
        sample_indices: 要详细分析的样本索引列表
    """
    print(f"\n{'='*80}")
    print(f"差异分析")
    print(f"{'='*80}")
    
    # 确保形状一致
    min_T = min(spu_scores.shape[0], gpu_scores.shape[0])
    spu_scores = spu_scores[:min_T]
    gpu_scores = gpu_scores[:min_T]
    
    print(f"\nSPU scores shape: {spu_scores.shape}")
    print(f"GPU scores shape: {gpu_scores.shape}")
    
    # 转换为 numpy 便于分析 (先转为 float32，因为 numpy 不支持 bfloat16)
    spu_np = spu_scores.float().cpu().numpy()
    gpu_np = gpu_scores.float().cpu().numpy()
    
    # 1. 全局统计
    print(f"\n{'='*60}")
    print(f"1. 全局统计")
    print(f"{'='*60}")
    
    diff = spu_np - gpu_np
    abs_diff = np.abs(diff)
    
    print(f"\n绝对差异统计:")
    print(f"  Mean:   {abs_diff.mean():.6f}")
    print(f"  Std:    {abs_diff.std():.6f}")
    print(f"  Min:    {abs_diff.min():.6f}")
    print(f"  Max:    {abs_diff.max():.6f}")
    print(f"  Median: {np.median(abs_diff):.6f}")
    print(f"  95%:    {np.percentile(abs_diff, 95):.6f}")
    print(f"  99%:    {np.percentile(abs_diff, 99):.6f}")
    
    print(f"\n相对差异统计 (abs_diff / (abs(gpu) + 1e-8)):")
    gpu_abs = np.abs(gpu_np) + 1e-8
    rel_diff = abs_diff / gpu_abs
    print(f"  Mean:   {rel_diff.mean():.6f}")
    print(f"  Median: {np.median(rel_diff):.6f}")
    print(f"  95%:    {np.percentile(rel_diff, 95):.6f}")
    
    # MSE & Cosine similarity
    mse = np.mean((spu_np - gpu_np) ** 2)
    print(f"\nMSE: {mse:.6e}")
    
    # Cosine similarity (flatten)
    spu_flat = spu_np.reshape(-1)
    gpu_flat = gpu_np.reshape(-1)
    cos_sim = np.dot(spu_flat, gpu_flat) / (np.linalg.norm(spu_flat) * np.linalg.norm(gpu_flat))
    print(f"Cosine Similarity: {cos_sim:.6f}")
    
    # 2. 按时间步统计
    print(f"\n{'='*60}")
    print(f"2. 按时间步统计 (Time-wise)")
    print(f"{'='*60}")
    
    time_mse = np.mean((spu_np - gpu_np) ** 2, axis=(1, 2))
    print(f"\n每个时间步的 MSE:")
    print(f"  Mean:   {time_mse.mean():.6e}")
    print(f"  Std:    {time_mse.std():.6e}")
    print(f"  Min:    {time_mse.min():.6e} (t={time_mse.argmin()})")
    print(f"  Max:    {time_mse.max():.6e} (t={time_mse.argmax()})")
    
    # 3. 按样本统计
    print(f"\n{'='*60}")
    print(f"3. 按样本统计 (Sample-wise)")
    print(f"{'='*60}")
    
    sample_mse = np.mean((spu_np - gpu_np) ** 2, axis=(0, 2))
    print(f"\n每个样本的 MSE:")
    print(f"  Mean:   {sample_mse.mean():.6e}")
    print(f"  Std:    {sample_mse.std():.6e}")
    print(f"  Min:    {sample_mse.min():.6e} (sample={sample_mse.argmin()})")
    print(f"  Max:    {sample_mse.max():.6e} (sample={sample_mse.argmax()})")
    
    # 显示最差的几个样本
    worst_samples = np.argsort(sample_mse)[-5:][::-1]
    print(f"\nMSE 最大的 5 个样本:")
    for rank, idx in enumerate(worst_samples, 1):
        print(f"  {rank}. Sample {idx}: MSE = {sample_mse[idx]:.6e}")
    
    # 4. 详细分析指定样本
    if sample_indices is not None:
        print(f"\n{'='*60}")
        print(f"4. 详细样本分析")
        print(f"{'='*60}")
        
        for idx in sample_indices:
            if idx >= spu_np.shape[1]:
                print(f"\n样本 {idx} 超出范围,跳过")
                continue
            
            print(f"\n样本 {idx}:")
            spu_sample = spu_np[:, idx, :]
            gpu_sample = gpu_np[:, idx, :]
            
            sample_diff = spu_sample - gpu_sample
            sample_abs_diff = np.abs(sample_diff)
            
            print(f"  Shape: {spu_sample.shape}")
            print(f"  SPU range: [{spu_sample.min():.4f}, {spu_sample.max():.4f}]")
            print(f"  GPU range: [{gpu_sample.min():.4f}, {gpu_sample.max():.4f}]")
            print(f"  Abs diff mean: {sample_abs_diff.mean():.6f}")
            print(f"  Abs diff max:  {sample_abs_diff.max():.6f}")
            print(f"  MSE: {np.mean(sample_diff ** 2):.6e}")
            
            # 找出差异最大的时间步和类别
            max_pos = np.unravel_index(sample_abs_diff.argmax(), sample_abs_diff.shape)
            print(f"  最大差异位置: t={max_pos[0]}, class={max_pos[1]}")
            print(f"    SPU value: {spu_sample[max_pos]:.6f}")
            print(f"    GPU value: {gpu_sample[max_pos]:.6f}")
            print(f"    Difference: {sample_diff[max_pos]:.6f}")
    
    return {
        'mse': mse,
        'mean_abs_diff': abs_diff.mean(),
        'max_abs_diff': abs_diff.max(),
        'cosine_similarity': cos_sim,
        'sample_mse': sample_mse,
    }


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='分析 SPU 和 GPU forward 输出差异')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据目录 (包含 .pt 文件)')
    parser.add_argument('--model', type=str, default='lstm_ctc_crf',
                        help='模型配置名')
    parser.add_argument('--pre_trained_params_file', type=str, default='',
                        help='预训练模型权重文件')
    parser.add_argument('--file_id', type=str, default=None,
                        help='数据文件 ID (如 0, 1, 2...), 不指定则使用第一个可用文件')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='设备 (如 cuda:0, cpu)')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='分析的样本数量 (从每个文件的512个样本中选取)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='GPU forward 的 batch size')
    parser.add_argument('--use_quant', action='store_true',
                        help='使用量化模型 (加载 act_scales 和 io_quant 权重)')
    parser.add_argument('--act_scales_path', type=str, 
                        default='/workspace/huada/scall/caoyu/layer_9_6x_act_scales_0305.pth',
                        help='激活量化 scales 文件路径')
    parser.add_argument('--io_quant_path', type=str,
                        default='/workspace/huada/scall/caoyu/layer_9_6x_io_quant_0305.pth',
                        help='量化权重文件路径')
    parser.add_argument('--bitwidth', type=int, default=8,
                        help='量化位宽')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("警告: CUDA 不可用,使用 CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"使用设备: {device}")
    
    # 列出可用文件
    available_files = list_available_files(args.data_dir)
    if not available_files:
        print(f"错误: 在 {args.data_dir} 中没有找到有效的数据文件")
        return
    
    print(f"\n找到 {len(available_files)} 个数据文件:")
    for fid, has_raw in available_files[:10]:  # 只显示前10个
        status = "✓ 有原始数据" if has_raw else "✗ 无原始数据"
        print(f"  文件 {fid}: {status}")
    if len(available_files) > 10:
        print(f"  ... (还有 {len(available_files) - 10} 个文件)")
    
    # 选择文件
    if args.file_id is None:
        # 使用第一个有原始数据的文件
        file_id = None
        for fid, has_raw in available_files:
            if has_raw:
                file_id = fid
                break
        
        if file_id is None:
            print("错误: 没有找到包含原始数据的文件")
            return
    else:
        file_id = args.file_id
        # 检查文件是否存在
        found = False
        has_raw = False
        for fid, hr in available_files:
            if fid == file_id:
                found = True
                has_raw = hr
                break
        if not found:
            print(f"错误: 文件 ID {file_id} 不存在")
            return
        if not has_raw:
            print(f"错误: 文件 ID {file_id} 没有原始数据")
            return
    
    print(f"\n使用文件 ID: {file_id}")
    
    # 加载数据
    spu_backbone_output, raw_data, targets, lengths = load_data_sample(
        args.data_dir, file_id
    )
    
    if raw_data is None:
        print("错误: 原始数据不存在,无法进行对比")
        return
    
    # 加载模型
    print(f"\n{'='*80}")
    print(f"加载模型")
    print(f"{'='*80}")
    
    config_file = f"{pro_dir}/opencall/configs/{args.model}.toml"
    print(f"配置文件: {config_file}")
    
    if args.use_quant:
        # 使用量化模型
        print(f"模式: 量化模型")
        print(f"Act scales: {args.act_scales_path}")
        print(f"IO quant: {args.io_quant_path}")
        print(f"量化位宽: {args.bitwidth}")
        
        # 检查文件是否存在
        if not os.path.exists(args.act_scales_path):
            print(f"错误: Act scales 文件不存在: {args.act_scales_path}")
            return
        if not os.path.exists(args.io_quant_path):
            print(f"错误: IO quant 文件不存在: {args.io_quant_path}")
            return
        
        print(f"提示: 正在创建模型架构... (这可能需要1-2分钟)")
        start_time = time.time()
        model = network(config_file)
        init_time = time.time() - start_time
        print(f"模型架构创建完成 (耗时: {init_time:.1f}秒)")
        
        # 替换 LSTM 为 FastLSTM (在加载权重之前)
        print(f"\n正在替换 torch.nn.LSTM 为 FastLSTM...")
        model = replace_lstm_with_manual(model)
        
        # 插入 FakeQuant 层
        print(f"\n正在插入 FakeQuant 层...")
        act_scales = torch.load(args.act_scales_path, map_location='cpu')
        model = insert_fakequant_encoder(model, act_scales, args.bitwidth, 'cpu')
        
        # 加载量化权重
        print(f"\n正在加载量化权重...")
        state_dict = torch.load(args.io_quant_path, map_location='cpu')
        state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
        model.load_state_dict(state_dict)
        print(f"量化权重加载完成")
        
        # 移动到目标设备并转换为 bfloat16
        print(f"\n正在将模型移动到设备: {device}")
        model = model.to(device)
        print(f"正在转换模型为 bfloat16...")
        model = model.to(torch.bfloat16)
        print(f"模型准备完成 (dtype: bfloat16)")
        
    elif args.pre_trained_params_file and os.path.exists(args.pre_trained_params_file):
        # 使用普通预训练模型
        print(f"模式: 普通预训练模型")
        print(f"加载预训练权重: {args.pre_trained_params_file}")
        print(f"提示: 正在创建模型架构... (这可能需要1-2分钟)")
        
        # 先在 CPU 上创建模型
        start_time = time.time()
        model = network(config_file)
        init_time = time.time() - start_time
        print(f"模型架构创建完成 (耗时: {init_time:.1f}秒)")
        
        # 替换 LSTM 为 FastLSTM (在加载权重之前)
        print(f"\n正在替换 torch.nn.LSTM 为 FastLSTM...")
        model = replace_lstm_with_manual(model)
        
        # 直接加载权重到 CPU
        print(f"正在加载预训练权重...")
        state_dict = torch.load(args.pre_trained_params_file, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"权重加载完成")
        
        # 再移动到目标设备并转换为 bfloat16
        print(f"正在将模型移动到设备: {device}")
        model = model.to(device)
        print(f"正在转换模型为 bfloat16...")
        model = model.to(torch.bfloat16)
        print(f"模型准备完成 (dtype: bfloat16)")
    else:
        # 使用随机初始化
        print("模式: 随机初始化")
        print("警告: 未提供预训练权重,将进行随机初始化...")
        print("提示: 模型初始化可能需要较长时间 (正交初始化)")
        model = network(config_file)
        
        # 替换 LSTM 为 FastLSTM
        print(f"\n正在替换 torch.nn.LSTM 为 FastLSTM...")
        model = replace_lstm_with_manual(model)
        
        # 移动到设备并转换为 bfloat16
        model = model.to(device)
        print(f"正在转换模型为 bfloat16...")
        model = model.to(torch.bfloat16)
        print(f"模型准备完成 (dtype: bfloat16)")
    
    model.eval()
    
    # 显示模型结构信息
    print(f"\n{'='*80}")
    print(f"模型结构信息")
    print(f"{'='*80}")
    encoder = model.encoder
    print(f"Encoder 类型: {type(encoder).__name__}")
    print(f"Encoder 层数: {len(list(encoder._modules.values()))}")
    for i, (name, layer) in enumerate(encoder._modules.items()):
        layer_type = type(layer).__name__
        # 如果是 LSTM 包装器，显示内部 rnn 的类型
        if hasattr(layer, 'rnn'):
            rnn_type = type(layer.rnn).__name__
            print(f"  Layer {i} ({name}): {layer_type} (内部: {rnn_type})")
        else:
            print(f"  Layer {i} ({name}): {layer_type}")
    print(f"\n说明: Backbone = encoder[:-1] (前 {len(list(encoder._modules.values()))-1} 层)")
    print(f"      LinearCRFEncoder = encoder[-1] (最后 1 层)")
    
    # GPU forward (原始数据 -> backbone, 使用 FastLSTM bf16)
    print(f"\n{'='*80}")
    print(f"GPU 路径 (FastLSTM bf16): 原始数据 -> Backbone")
    print(f"{'='*80}")
    print(f"输入: 原始数据 {raw_data.shape}")
    print(f"Batch size: {args.batch_size}")
    print(f"说明: 模型使用 FastLSTM (activation_impl='moffett', dtype=bfloat16)")
    
    gpu_scores = gpu_forward(model, raw_data, device, batch_size=args.batch_size)
    print(f"输出 (Backbone): {gpu_scores.shape}, dtype={gpu_scores.dtype}")
    print(f"  Range: [{gpu_scores.min().item():.4f}, {gpu_scores.max().item():.4f}]")
    print(f"  Mean: {gpu_scores.mean().item():.4f}")
    print(f"  Std: {gpu_scores.std().item():.4f}")
    
    # SPU forward (直接使用预计算的 backbone 输出)
    print(f"\n{'='*80}")
    print(f"SPU 路径: 直接使用预计算的 Backbone 输出")
    print(f"{'='*80}")
    print(f"输入: SPU backbone 预计算输出 {spu_backbone_output.shape}")
    
    spu_scores = spu_forward(spu_backbone_output, device)
    print(f"输出 (Backbone): {spu_scores.shape}, dtype={spu_scores.dtype}")
    print(f"  Range: [{spu_scores.min().item():.4f}, {spu_scores.max().item():.4f}]")
    print(f"  Mean: {spu_scores.mean().item():.4f}")
    print(f"  Std: {spu_scores.std().item():.4f}")
    
    # 分析差异
    # 选择一些样本进行详细分析
    N = spu_scores.shape[1]
    sample_indices = np.linspace(0, N-1, min(args.num_samples, N), dtype=int).tolist()
    
    results = analyze_difference(spu_scores, gpu_scores, sample_indices=sample_indices)
    
    # 保存结果
    print(f"\n{'='*80}")
    print(f"保存分析结果")
    print(f"{'='*80}")
    
    output_dir = "/workspace/huada/scall"
    result_file = os.path.join(output_dir, f"spu_gpu_diff_analysis_{file_id}.txt")
    
    with open(result_file, 'w') as f:
        f.write(f"SPU vs GPU Backbone 输出差异分析\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"说明: 对比 SPU 和 GPU(FastLSTM bf16) 的 Backbone 输出\n\n")
        f.write(f"数据文件 ID: {file_id}\n")
        f.write(f"模型: {args.model}\n")
        if args.use_quant:
            f.write(f"模式: 量化模型\n")
            f.write(f"Act scales: {args.act_scales_path}\n")
            f.write(f"IO quant: {args.io_quant_path}\n")
        else:
            f.write(f"预训练权重: {args.pre_trained_params_file}\n")
        f.write(f"\n")
        f.write(f"SPU backbone output shape: {spu_scores.shape}\n")
        f.write(f"GPU backbone output shape: {gpu_scores.shape}\n\n")
        f.write(f"MSE: {results['mse']:.6e}\n")
        f.write(f"Mean Abs Diff: {results['mean_abs_diff']:.6f}\n")
        f.write(f"Max Abs Diff: {results['max_abs_diff']:.6f}\n")
        f.write(f"Cosine Similarity: {results['cosine_similarity']:.6f}\n")
    
    print(f"分析结果已保存到: {result_file}")
    
    # 可选: 保存 backbone outputs 用于进一步分析 (转换为 float32 以便兼容)
    scores_file = os.path.join(output_dir, f"spu_gpu_backbone_outputs_{file_id}.pt")
    torch.save({
        'spu_backbone_output': spu_scores.float().cpu(),
        'gpu_backbone_output': gpu_scores.float().cpu(),
        'file_id': file_id,
    }, scores_file)
    print(f"Backbone outputs 已保存到: {scores_file}")


if __name__ == "__main__":
    main()
