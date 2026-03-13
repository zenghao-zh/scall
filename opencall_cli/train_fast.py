"""
高效分布式训练脚本 - 重构版本

优化特性:
1. 梯度累积 - 支持更大的有效 batch size
2. torch.compile - PyTorch 2.0+ 模型编译加速
3. 混合精度训练 - AMP + GradScaler
4. 异步数据预取 - CUDA Stream 流水线
5. 断点续训 - 完整的训练状态保存/恢复
6. 进度条显示 - 使用 tqdm
7. 性能统计 - 吞吐量跟踪
8. 智能验证 - 支持 early stopping

Usage:
    torchrun --nproc_per_node=4 train_fast.py --data_dir /path/to/data --output_name exp1
"""

import sys
import os
pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pro_dir)

import glob
import re
import json
import time
import argparse
from datetime import timedelta
from contextlib import nullcontext
from typing import Optional, Dict, Any, Tuple

import numpy as np
import random
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# 尝试导入 tqdm，如果不存在则使用简单的进度显示
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from opencall.utils.util import (
    model_eval, init, network, 
    get_dataset_in_one_dir, get_dataset_from_pt, 
    get_lr_scheduler, log_func
)
from pruning import PrunerScheduler


# ============================================================================
# 辅助类和函数
# ============================================================================

def clip_gradient(optimizer, grad_clip):
    """
    与原始代码保持一致的梯度裁剪函数
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def fake_quantize_int8(weight):
    """INT8 per-tensor symmetric fake quantization with STE.
    
    Forward: returns quantize-then-dequantize value (simulates INT8 precision loss).
    Backward: gradient passes through to the original weight (straight-through estimator).
    """
    amax = weight.abs().max()
    if amax == 0:
        return weight
    scale = amax / 127.0
    weight_q = torch.clamp(torch.round(weight / scale), -127, 127)
    # STE trick: (quantized - original).detach() + original
    # Forward uses quantized value; backward gradient flows to original weight
    return (weight_q * scale - weight).detach() + weight


def fake_quantize_int8_detached(weight):
    """INT8 per-tensor symmetric fake quantization (no gradient, for inference/saving)."""
    amax = weight.abs().max()
    if amax == 0:
        return weight.clone()
    scale = amax / 127.0
    weight_q = torch.clamp(torch.round(weight / scale), -127, 127)
    return weight_q * scale


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DataPrefetcher:
    """
    CUDA 数据预取器 - 使用 CUDA Stream 实现异步数据传输
    
    在 GPU 计算当前 batch 的同时，预取下一个 batch 到 GPU 内存
    """
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.iter = None
        self.batch = None
        
    def __iter__(self):
        self.iter = iter(self.loader)
        self.preload()
        return self
    
    def preload(self):
        try:
            self.batch = next(self.iter)
        except StopIteration:
            self.batch = None
            return
        
        with torch.cuda.stream(self.stream):
            if isinstance(self.batch, (list, tuple)):
                self.batch = tuple(
                    b.to(self.device, non_blocking=True) if isinstance(b, torch.Tensor) else b 
                    for b in self.batch
                )
            elif isinstance(self.batch, torch.Tensor):
                self.batch = self.batch.to(self.device, non_blocking=True)
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
    
    def __len__(self):
        return len(self.loader)


class CheckpointManager:
    """检查点管理器 - 支持断点续训"""
    
    def __init__(self, save_dir: str, max_keep: int = 3):
        self.save_dir = save_dir
        self.max_keep = max_keep
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, state: Dict[str, Any], epoch: int, step: int, is_best: bool = False):
        """保存检查点"""
        # 保存当前检查点
        ckpt_path = os.path.join(self.save_dir, f"checkpoint_epoch{epoch}_step{step}.pt")
        torch.save(state, ckpt_path)
        
        # 保存最新检查点的链接
        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        torch.save(state, latest_path)
        
        # 如果是最佳模型，额外保存
        if is_best:
            best_path = os.path.join(self.save_dir, "checkpoint_best.pt")
            torch.save(state, best_path)
        
        # 清理旧的检查点
        self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """保留最近的 max_keep 个检查点"""
        ckpts = glob.glob(os.path.join(self.save_dir, "checkpoint_epoch*.pt"))
        if len(ckpts) > self.max_keep:
            # 按修改时间排序
            ckpts.sort(key=os.path.getmtime)
            for ckpt in ckpts[:-self.max_keep]:
                os.remove(ckpt)
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """加载最新的检查点"""
        latest_path = os.path.join(self.save_dir, "checkpoint_latest.pt")
        if os.path.exists(latest_path):
            return torch.load(latest_path, map_location='cpu')
        return None
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        best_path = os.path.join(self.save_dir, "checkpoint_best.pt")
        if os.path.exists(best_path):
            return torch.load(best_path, map_location='cpu')
        return None


# ============================================================================
# 训练器类
# ============================================================================

class Trainer:
    """高效训练器"""
    
    def __init__(self, args):
        self.args = args
        self.local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
        self.global_rank = None
        self.world_size = None
        
        # 训练状态
        self.epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # 组件
        self.model = None
        self.model_orig = None
        self.optimizer = None
        self.lr_scheduler = None
        self.scaler = None
        self.pruner = None
        self.train_loader = None
        self.valid_loader = None
        self.log_writer = None
        self.ckpt_manager = None
        
        # 路径
        self.res_dir = f'/workspace/huada/task_results/{args.output_name}'
        self.log_path = f"{self.res_dir}/training.log"
        self.config_file_path = f"{pro_dir}/opencall/configs/{args.model}.toml"
        
        # 性能统计
        self.loss_meter = AverageMeter()
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.throughput_meter = AverageMeter()
    
    def setup(self):
        """初始化所有组件"""
        self._setup_distributed()
        self._setup_seed()
        self._setup_directories()
        self._setup_data()
        self._setup_model()
        if self.args.qat_int8:
            self._load_qat_weights()  # QAT: 先加载权重, 再初始化 optimizer
        self._setup_optimizer()
        self._setup_checkpoint()
        self._try_resume()  # 必须在 pruner 之前恢复模型/优化器状态
        if self.args.qat_int8:
            self._setup_qat()  # QAT 模式: 注册 fake quantization hooks
        else:
            self._setup_pruner()  # pruner 需要使用正确的模型状态
    
    def _setup_distributed(self):
        """初始化分布式训练"""
        torch.cuda.set_device(self.local_rank)
        # 增加 timeout 以支持长时间的验证阶段
        # 验证可能需要 1-2 小时，所以设置为 3 小时
        timeout = timedelta(hours=3)
        dist.init_process_group(backend="nccl", timeout=timeout)
        self.global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        if self.is_main_process:
            print(f"[Distributed] World size: {self.world_size}")
    
    def _setup_seed(self):
        """设置随机种子和 CUDA 优化"""
        # 与原始代码保持一致: 使用固定种子 1
        torch_seed = 1
        torch.manual_seed(torch_seed)
        torch.cuda.manual_seed(torch_seed)
        torch.cuda.manual_seed_all(torch_seed)
        np.random.seed(torch_seed)
        random.seed(torch_seed)
        
        # 与原始代码保持一致的 cudnn 设置
        # deterministic=True 确保训练可复现且稳定
        # benchmark=False 避免算法选择导致的数值波动
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # 调用原始的 init 函数 (设置额外的随机种子)
        init(self.args.seed, self.local_rank, (not self.args.nondeterministic))
    
    def _setup_directories(self):
        """创建输出目录"""
        if self.is_main_process:
            os.makedirs(self.res_dir, exist_ok=True)
            if os.path.exists(self.config_file_path):
                os.system(f"cp {self.config_file_path} {self.res_dir}/config.toml")
            self.log_writer = SummaryWriter(os.path.join(self.res_dir, "logs"))
            
            # 记录训练参数
            self._log(f"{'=' * 20} START TRAINING {'=' * 20}")
            for key, value in vars(self.args).items():
                self._log(f"{key}: {value}")
    
    def _setup_data(self):
        """加载数据"""
        if self.is_main_process:
            self._log("[Loading data]")
        
        if self.args.data_type == "pt":
            self.train_loader, self.valid_loader = get_dataset_from_pt(self.args)
        else:
            self.train_loader, self.valid_loader = get_dataset_in_one_dir(self.args)
        
        self.data_len = len(self.train_loader)
        
        if self.is_main_process:
            self._log(f"Train batches per epoch: {self.data_len}")
            self._log(f"Effective batch size: {self.args.batch_size * self.world_size * self.args.grad_accum}")
    
    def _setup_model(self):
        """初始化模型"""
        self.model_orig = network(self.config_file_path).to(self.local_rank)
        
        # 加载预训练权重
        if os.path.exists(self.args.pre_trained_params_file):
            if self.is_main_process:
                print(f"Loading pretrained model: {self.args.pre_trained_params_file}")
            self.model_orig.load_state_dict(
                torch.load(self.args.pre_trained_params_file, map_location=f'cuda:{self.local_rank}')
            )
        
        # torch.compile 加速 (PyTorch 2.0+)
        if self.args.compile and hasattr(torch, 'compile'):
            if self.is_main_process:
                print("[Compiling model with torch.compile]")
            self.model_orig = torch.compile(self.model_orig, mode="reduce-overhead")
        
        # 包装为 DDP
        # 注意：与原始代码保持一致，不使用 static_graph 和 gradient_as_bucket_view
        # 这些优化可能导致某些模型训练不稳定
        find_unused = "conformer" in self.args.model
        self.model = DDP(
            self.model_orig, 
            device_ids=[self.local_rank], 
            output_device=self.local_rank,
            find_unused_parameters=find_unused,
        )
        
        if self.is_main_process:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self._log(f"Total parameters: {total_params:,}")
            self._log(f"Trainable parameters: {trainable_params:,}")
    
    def _setup_optimizer(self):
        """初始化优化器和学习率调度器"""
        # 稀疏的改成0，其它的不变
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.lr,
            weight_decay=1e-2,
        )
        
        # Linear Warmup + Cosine Decay (与 train_index_ddp_sparse.py 一致)
        effective_steps_per_epoch = self.data_len // self.args.grad_accum
        total_steps = effective_steps_per_epoch * self.args.epoch_num
        self.lr_scheduler = get_lr_scheduler(
            epochs=total_steps,
            optimizer=self.optimizer,
            data_len=1,
            last_epoch=0,
            end_ratio=0.01,
            warmup_steps=self.args.warmup_steps,
        )
        
        if self.is_main_process:
            print(f"[LR Scheduler] Linear Warmup + Cosine Decay: "
                  f"lr={self.args.lr}, end_ratio=0.01, "
                  f"warmup={self.args.warmup_steps}, total_steps={total_steps}")
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
    
    def _setup_pruner(self):
        """初始化剪枝器"""
        if self.args.pruning != 1:
            return
        
        # 注意：使用梯度累积时，实际的参数更新次数 = data_len / grad_accum
        # pruner 的 step 是基于参数更新次数，而非 batch 次数
        effective_steps_per_epoch = self.data_len // self.args.grad_accum
        total_steps = effective_steps_per_epoch * self.args.epoch_num
        
        sparsity_list = [self.args.sparsity]
        prune_log = os.path.join(self.res_dir, f"{self.args.prune_log}{self.args.epoch_num}.log")
        
        if self.is_main_process:
            print(f"[Pruner] Steps per epoch: {effective_steps_per_epoch}, Total steps: {total_steps}")
        
        prune_dict = {
            "encoder.0.conv.weight": 0,
            "encoder.1.conv.weight": 0,
            "encoder.2.conv.weight": 0,
        }
        
        # 设置最后一个 linear 层的稀疏率
        if self.args.last_linear_sparsity is not None:
            # 动态查找最后一个 linear.weight 层的名字
            last_linear_name = None
            for name, _ in self.model_orig.named_parameters():
                if 'linear.weight' in name:
                    last_linear_name = name
            if last_linear_name is not None:
                prune_dict[last_linear_name] = self.args.last_linear_sparsity
                if self.is_main_process:
                    if self.args.last_linear_sparsity == 0:
                        print(f"[Pruner] Last linear '{last_linear_name}' excluded from pruning")
                    else:
                        multiplier = 1.0 / (1.0 - self.args.last_linear_sparsity) if self.args.last_linear_sparsity < 1.0 else float('inf')
                        print(f"[Pruner] Last linear '{last_linear_name}' sparsity={self.args.last_linear_sparsity:.4f} ({multiplier:.1f}x)")
                
                # Resume 时从高稀疏率 → 低稀疏率：重新初始化被剪枝的零权重
                # 否则 BBS 的 mask 机制会让零权重永远无法恢复
                if self.args.resume and 0 < self.args.last_linear_sparsity < 1.0:
                    self._reinit_pruned_weights(last_linear_name, self.args.last_linear_sparsity)
        
        # 检查是否有 pruner 检查点需要恢复
        pruner_resume_dict = None
        pruner_ckpt_path = os.path.join(prune_log, 'pruner_scheduler.pth')
        if self.args.resume and os.path.exists(pruner_ckpt_path):
            if self.is_main_process:
                print(f"[Resume] Found pruner checkpoint: {pruner_ckpt_path}")
            pruner_resume_dict = {
                'path': pruner_ckpt_path,
                'load_model_states': False,  # 模型状态已经从主检查点恢复
                'load_optimizer_states': False,  # 优化器状态已经从主检查点恢复
            }
        
        self.pruner = PrunerScheduler(
            self.model.module,
            optimizer=self.optimizer,
            prune_dict=prune_dict,
            steps_per_epoch=effective_steps_per_epoch,
            num_steps=total_steps,
            prune_freq=500,
            log_freq=500,
            seq_len=4796,
            sparsities=sparsity_list,
            rank=self.global_rank,
            bank_size=64,
            log_path=prune_log,
            finetune=False,
            pruner_resume_dict=pruner_resume_dict,
        )
        
        # 修复：pruner_resume_dict 路径中 PrunerScheduler.__init__ 内部
        # load_state_dict 会把旧 base_lrs 恢复，这里统一覆盖为当前 LR
        if pruner_resume_dict is not None:
            idx = self.pruner.index
            self.pruner.optim_schedulers[idx].base_lrs = [
                group['initial_lr'] for group in self.optimizer.param_groups
            ]
            if self.is_main_process:
                print(f"[Resume] Fixed pruner scheduler base_lrs to {self.pruner.optim_schedulers[idx].base_lrs}")
        
        # Resume 时将 global_step 同步给 pruner
        # PrunerScheduler.step 和 Trainer.global_step 是 1:1 递增的，
        # 但 pruner 没有独立的 save/load，所以利用已保存的 global_step 恢复
        if self.args.resume and self.global_step > 0 and pruner_resume_dict is None:
            self.pruner.step = self.global_step
            self.pruner.resume_tag = True
            self.pruner.init_pruner()
            
            # 修复：将 pruner 的 LR scheduler 推进到正确的位置
            # 不推进的话，scheduler 从 step 0 重新开始，会在 fine-tune 阶段返回 1.0（全量 LR），
            # 导致恢复训练后 LR 恒定为 initial_lr，loss 无法下降
            idx = self.pruner.index
            steps_into_stage = self.global_step - self.pruner.stage_wise_steps[idx][0]
            
            # 优先从主检查点中恢复 pruner scheduler 状态
            resumed_from_ckpt = False
            if hasattr(self, '_resume_pruner_state') and self._resume_pruner_state is not None:
                try:
                    saved_state = self._resume_pruner_state
                    if saved_state['index'] == idx:
                        self.pruner.optim_schedulers[idx].load_state_dict(
                            saved_state['lr_scheduler_state']
                        )
                        # 修复：load_state_dict 会把旧 checkpoint 的 base_lrs 一并恢复，
                        # 如果 resume 时换了 LR（如从 5e-4 改为 4e-6），base_lrs 会被污染。
                        # 这里强制用当前 optimizer 的 initial_lr 覆盖。
                        self.pruner.optim_schedulers[idx].base_lrs = [
                            group['initial_lr'] for group in self.optimizer.param_groups
                        ]
                        resumed_from_ckpt = True
                        if self.is_main_process:
                            print(f"[Resume] Restored pruner LR scheduler state from main checkpoint")
                except Exception as e:
                    if self.is_main_process:
                        print(f"[Resume] Could not restore pruner scheduler state ({e}), advancing manually")
            
            # 若无法从检查点恢复，则手动推进 scheduler 步数
            if not resumed_from_ckpt and steps_into_stage > 0:
                for _ in range(steps_into_stage):
                    self.pruner.optim_schedulers[idx].step()
                if self.is_main_process:
                    print(f"[Resume] Manually advanced pruner LR scheduler by {steps_into_stage} steps")
            
            if self.is_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                stage_info = self.pruner.stage_wise_steps[idx]
                prune_end = stage_info[1]
                finetune_end = stage_info[2]
                phase = "pruning" if self.global_step < prune_end else "fine-tuning"
                print(f"[Resume] Synced pruner step to global_step={self.global_step}")
                print(f"[Resume] Stage [{stage_info[0]}, prune_end={prune_end}, total_end={finetune_end}], "
                      f"phase={phase}, steps_into_stage={steps_into_stage}")
                print(f"[Resume] Pruner LR after sync: {current_lr:.2e}")
    
    def _load_qat_weights(self):
        """加载 QAT 预训练权重.
        
        必须在 _setup_optimizer() 之前调用, 确保 optimizer 初始化时看到正确的权重,
        避免 optimizer 内部状态 (momentum, variance) 与实际权重不匹配.
        """
        if self.args.qat_weight_path and os.path.exists(self.args.qat_weight_path):
            if self.is_main_process:
                self._log(f"[QAT] Loading pretrained weights from: {self.args.qat_weight_path}")
            state_dict = torch.load(self.args.qat_weight_path, map_location=f'cuda:{self.local_rank}')
            self.model_orig.load_state_dict(state_dict)
            if self.is_main_process:
                self._log("[QAT] Pretrained weights loaded successfully")
        elif self.is_main_process:
            self._log("[QAT] Warning: qat_weight_path not set or not found, using current model weights")

    def _setup_qat(self):
        """注册 INT8 weight QAT forward hooks.
        
        对 LSTM weights 和 LinearCRFEncoder 的 linear weight 做 INT8 fake quantization.
        训练时通过 save/restore 方式实现 STE, 保证梯度回传到原始 FP32 权重.
        
        实现原理:
          - forward pre-hook 中将原始 FP32 权重备份, 替换为量化值用于 forward
          - forward post-hook 中将原始 FP32 权重恢复
          - backward 时梯度作用在原始 FP32 权重上 (等效 STE)
          
        注意: 不能在 hook 中用 fake_quantize_int8 的 STE trick + .data= 赋值,
        因为 .data= 脱离计算图, autograd 看不到, STE 梯度完全失效,
        且原始 FP32 权重会被永久覆盖为量化值, 精度不断截断累积.
        """
        # 收集需要量化的模块
        from opencall.models.common.nn import LSTM as LSTM_Module, LinearCRFEncoder
        
        qat_targets = []
        for name, module in self.model_orig.named_modules():
            if isinstance(module, LSTM_Module):
                qat_targets.append((name, module, 'lstm'))
            elif isinstance(module, LinearCRFEncoder):
                qat_targets.append((name, module, 'linear_crf'))
        
        if self.is_main_process:
            self._log(f"[QAT] Found {len(qat_targets)} modules to quantize:")
            for name, _, mtype in qat_targets:
                self._log(f"  - {name} ({mtype})")
        
        # 注册 forward pre-hooks + post-hooks (save/restore 方式实现 STE)
        # pre-hook: 备份原始 FP32 权重, 替换为 INT8 量化值
        # post-hook: 恢复原始 FP32 权重, 使 backward 梯度作用在原始权重上
        self._qat_hooks = []
        for name, module, mtype in qat_targets:
            if mtype == 'lstm':
                def lstm_pre_hook(mod, inputs, module_name=name):
                    # 备份原始 FP32 权重
                    mod._qat_saved_weight_ih = mod.rnn.weight_ih_l0.data.clone()
                    mod._qat_saved_weight_hh = mod.rnn.weight_hh_l0.data.clone()
                    # 替换为 INT8 量化值 (detached, 用于 forward 计算)
                    mod.rnn.weight_ih_l0.data = fake_quantize_int8_detached(mod.rnn.weight_ih_l0.data)
                    mod.rnn.weight_hh_l0.data = fake_quantize_int8_detached(mod.rnn.weight_hh_l0.data)
                def lstm_post_hook(mod, inputs, outputs, module_name=name):
                    # 恢复原始 FP32 权重, 使 backward 梯度作用在原始权重上
                    mod.rnn.weight_ih_l0.data = mod._qat_saved_weight_ih
                    mod.rnn.weight_hh_l0.data = mod._qat_saved_weight_hh
                    del mod._qat_saved_weight_ih, mod._qat_saved_weight_hh
                self._qat_hooks.append(module.register_forward_pre_hook(lstm_pre_hook))
                self._qat_hooks.append(module.register_forward_hook(lstm_post_hook))
            elif mtype == 'linear_crf':
                def linear_pre_hook(mod, inputs, module_name=name):
                    mod._qat_saved_weight = mod.linear.weight.data.clone()
                    mod.linear.weight.data = fake_quantize_int8_detached(mod.linear.weight.data)
                def linear_post_hook(mod, inputs, outputs, module_name=name):
                    mod.linear.weight.data = mod._qat_saved_weight
                    del mod._qat_saved_weight
                self._qat_hooks.append(module.register_forward_pre_hook(linear_pre_hook))
                self._qat_hooks.append(module.register_forward_hook(linear_post_hook))
        
        if self.is_main_process:
            self._log(f"[QAT] Registered {len(self._qat_hooks)} hooks (pre+post) for INT8 fake quantization")
        
        # 记录初始稀疏 mask (True = 该位置是零, 需要保持稀疏)
        # optimizer step 后调用 _apply_sparsity_masks() 恢复这些位置为零
        self._sparsity_masks = {}
        total_sparse = 0
        total_params = 0
        for name, param in self.model_orig.named_parameters():
            mask = (param.data == 0)
            n_sparse = mask.sum().item()
            if n_sparse > 0:
                self._sparsity_masks[name] = mask
                total_sparse += n_sparse
            total_params += param.numel()
        
        if self.is_main_process:
            sparsity_ratio = total_sparse / total_params if total_params > 0 else 0
            self._log(f"[QAT] Sparsity masks captured: {len(self._sparsity_masks)} tensors, "
                      f"overall sparsity={sparsity_ratio:.4f} ({total_sparse:,}/{total_params:,})")

    def _apply_sparsity_masks(self):
        """将 optimizer step 后被更新的稀疏零位置重新置零."""
        if not hasattr(self, '_sparsity_masks'):
            return
        with torch.no_grad():
            for name, param in self.model_orig.named_parameters():
                if name in self._sparsity_masks:
                    param.data[self._sparsity_masks[name]] = 0.0
    
    def _reinit_pruned_weights(self, layer_name: str, target_sparsity: float):
        """
        当 resume 从高稀疏率 checkpoint 训练到低稀疏率时，
        重新初始化已经被剪枝为零的权重，让 BBS pruner 能正确工作。
        
        原因：BBS 在每个 prune step 先用旧 mask 把零权重置零，然后按 topk(abs) 选择保留权重。
        已经是零的权重绝对值为 0，永远不会被选中，导致稀疏率无法从高降到低。
        
        策略：用非零权重绝对值均值的 1% 作为初始化尺度，足够小不影响模型输出，
        又足够大让 pruner 可以"看到"这些权重并参与选择。
        """
        for name, param in self.model_orig.named_parameters():
            if name != layer_name:
                continue
            
            with torch.no_grad():
                weight = param.data
                zero_mask = (weight == 0)
                current_sparsity = zero_mask.float().mean().item()
                
                if current_sparsity <= target_sparsity:
                    if self.is_main_process:
                        print(f"[Reinit] '{name}' current sparsity {current_sparsity:.4f} "
                              f"<= target {target_sparsity:.4f}, no reinit needed")
                    return
                
                # 用非零权重的统计量初始化
                nonzero_weights = weight[~zero_mask]
                if nonzero_weights.numel() == 0:
                    if self.is_main_process:
                        print(f"[Reinit] WARNING: '{name}' has all-zero weights, using default init")
                    init_scale = 0.01
                else:
                    init_scale = nonzero_weights.abs().mean().item() * 0.01
                
                # 只重新初始化零权重位置
                reinit_values = torch.randn_like(weight) * init_scale
                weight[zero_mask] = reinit_values[zero_mask]
                
                new_sparsity = (weight == 0).float().mean().item()
                if self.is_main_process:
                    print(f"[Reinit] '{name}': sparsity {current_sparsity:.4f} → {new_sparsity:.4f} "
                          f"(target: {target_sparsity:.4f})")
                    print(f"[Reinit] Non-zero weight scale: mean_abs={nonzero_weights.abs().mean().item():.6f}, "
                          f"reinit_scale={init_scale:.6f}")
            break
    
    def _setup_checkpoint(self):
        """初始化检查点管理器"""
        self.ckpt_manager = CheckpointManager(self.res_dir, max_keep=3)
    
    def _try_resume(self):
        """尝试从检查点恢复"""
        if not self.args.resume:
            return
        
        ckpt = self.ckpt_manager.load_latest()
        if ckpt is None:
            if self.is_main_process:
                print("[Resume] No checkpoint found, starting from scratch")
            return
        
        self.epoch = ckpt['epoch']
        self.global_step = ckpt['global_step']
        self.best_metric = ckpt.get('best_metric', 0.0)
        self.model.module.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        
        # 保存 pruner 状态供 _setup_pruner 使用
        self._resume_pruner_state = ckpt.get('pruner_state', None)
        
        # 快进 lr_scheduler 到恢复位置
        if self.global_step > 0:
            for _ in range(self.global_step):
                self.lr_scheduler.step()
        
        if self.is_main_process:
            print(f"[Resume] Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
            print(f"[Resume] LR after resume: {self.optimizer.param_groups[0]['lr']:.2e}")
    
    @property
    def is_main_process(self) -> bool:
        return self.global_rank == 0
    
    def _log(self, msg: str):
        """记录日志"""
        if self.is_main_process:
            log_func(msg, self.log_path)
    
    def train(self):
        """主训练循环"""
        start_time = time.time()
        
        if self.is_main_process:
            print(f"[DEBUG] Entering train(), val_before_train={self.args.val_before_train}")
        
        # 训练开始前先执行一次验证（可选，用于确认初始状态）
        if self.args.val_before_train:
            if self.is_main_process:
                self._log("[Initial Validation] Before training starts")
                print("[DEBUG] Starting initial validation...")
                self._validate_only()
                print("[DEBUG] Initial validation done, waiting for barrier...")
            # 同步等待主进程完成初始验证
            dist.barrier()
            if self.is_main_process:
                print("[DEBUG] Barrier passed")
        
        if self.is_main_process:
            print(f"[DEBUG] Starting epoch loop, epochs={self.args.epoch_num}")
        
        for epoch in range(self.epoch, self.args.epoch_num):
            self.epoch = epoch
            if self.is_main_process:
                print(f"[DEBUG] Epoch {epoch}: calling set_epoch")
            self.train_loader.sampler.set_epoch(epoch)
            if self.is_main_process:
                print(f"[DEBUG] Epoch {epoch}: starting _train_epoch")
            
            # 训练一个 epoch
            epoch_loss = self._train_epoch()
            
            # epoch 结束时验证（只有主进程执行，与原始脚本一致）
            self._validate_and_save()
        
        # 训练结束，同步所有进程
        dist.barrier()
        
        total_time = time.time() - start_time
        if self.is_main_process:
            self._log(f"{'=' * 20} TRAINING FINISHED {'=' * 20}")
            self._log(f"Total training time: {total_time / 3600:.2f} hours")
            # 只有主进程执行最终评估
            self._final_evaluation()
        
        # 最终同步，确保主进程完成评估
        dist.barrier()
        
        # 清理分布式环境
        dist.destroy_process_group()
    
    def _train_epoch(self) -> float:
        """训练一个 epoch"""
        self.model.train()
        self.loss_meter.reset()
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.throughput_meter.reset()
        
        # 使用数据预取器
        if self.args.use_prefetch:
            data_iter = DataPrefetcher(self.train_loader, self.local_rank)
        else:
            data_iter = self.train_loader
        
        # 进度条 (仅主进程)
        if TQDM_AVAILABLE and self.is_main_process:
            pbar = tqdm(total=self.data_len, desc=f"Epoch {self.epoch}", dynamic_ncols=True)
        else:
            pbar = None
        
        end_time = time.time()
        
        for batch_idx, batch in enumerate(data_iter):
            data_time = time.time() - end_time
            self.data_time_meter.update(data_time)
            
            # 训练一个 step (支持梯度累积)
            loss = self._train_step(batch, batch_idx)
            
            # 更新统计
            batch_time = time.time() - end_time
            self.batch_time_meter.update(batch_time)
            self.loss_meter.update(loss)
            
            batch_size = batch[0].size(0) if isinstance(batch, (list, tuple)) else batch.size(0)
            throughput = batch_size * self.world_size / batch_time
            self.throughput_meter.update(throughput)
            
            # 日志
            if self.is_main_process and batch_idx % self.args.log_interval == 0:
                self._log_step(batch_idx)
            
            # 更新进度条
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{loss:.4f}',
                    'lr': f'{self._current_lr():.2e}',
                    'throughput': f'{throughput:.0f}'
                })
            
            # 检查是否达到限制
            if 0 < self.args.limit_train_size < batch_idx:
                break
            
            end_time = time.time()
        
        if pbar is not None:
            pbar.close()
        
        return self.loss_meter.avg
    
    def _train_step(self, batch, batch_idx: int) -> float:
        """训练一个 step，支持梯度累积"""
        grad_accum = self.args.grad_accum
        
        # 是否需要同步梯度 (梯度累积的最后一步才同步)
        is_accumulating = (batch_idx + 1) % grad_accum != 0
        
        # 与原始代码保持一致: 在每个 step 开始时清零梯度
        if batch_idx % grad_accum == 0:
            self.optimizer.zero_grad()
        
        # 混合精度上下文
        autocast_ctx = torch.cuda.amp.autocast(enabled=self.args.use_amp)
        
        # 梯度同步上下文 (累积时禁用同步以提高效率)
        if is_accumulating:
            sync_ctx = self.model.no_sync()
        else:
            sync_ctx = nullcontext()
        
        # 前向传播和反向传播
        with sync_ctx:
            with autocast_ctx:
                if isinstance(batch, (list, tuple)):
                    data, targets, lengths = batch
                    if not isinstance(data, torch.Tensor) or data.device.type != 'cuda':
                        data = data.to(self.local_rank, non_blocking=True)
                        targets = targets.to(self.local_rank, non_blocking=True)
                        lengths = lengths.to(self.local_rank, non_blocking=True)
                else:
                    data, targets, lengths = batch, None, None
                
                scores = self.model(data)
                losses = self.model_orig.loss(scores, targets, lengths)
                
                if not isinstance(losses, dict):
                    losses = {"loss": losses}
                
                loss = losses.get("total_loss", losses["loss"]) / grad_accum
            
            # 反向传播
            self.scaler.scale(loss).backward()
        
        # 梯度累积完成，更新参数
        if not is_accumulating:
            # 与原始代码保持一致的梯度裁剪顺序
            self.scaler.unscale_(self.optimizer)
            # 1. clip_grad_norm_ (max_norm=2.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            # 2. clip_gradient (clamp to [-1, 1])
            clip_gradient(self.optimizer, 1)
            
            # 优化器更新
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # QAT 稀疏保护: optimizer step 会把稀疏零位置更新为非零, 这里恢复
            if self.args.qat_int8:
                self._apply_sparsity_masks()
            
            # 学习率更新 & 剪枝 (与 train_index_ddp_sparse.py 一致：先 step scheduler，再 prune)
            self.lr_scheduler.step()
            if self.pruner is not None:
                self.pruner.prune()
            
            self.global_step += 1
            
            # Tensorboard
            if self.is_main_process and self.log_writer and self.global_step % 100 == 0:
                self.log_writer.add_scalar("train/loss", losses["loss"].item(), self.global_step)
                self.log_writer.add_scalar("train/lr", self._current_lr(), self.global_step)
                self.log_writer.add_scalar("train/throughput", self.throughput_meter.avg, self.global_step)
        
        return losses["loss"].item()
    
    def _current_lr(self) -> float:
        """获取当前实际学习率 (直接读 optimizer，避免 pruner/trainer 双 scheduler 不一致)"""
        return self.optimizer.param_groups[0]['lr']
    
    def _log_step(self, batch_idx: int):
        """记录训练步骤日志"""
        progress = batch_idx / self.data_len * 100
        lr = self._current_lr()
        
        msg = (
            f"Epoch [{self.epoch}/{self.args.epoch_num}] "
            f"Step [{batch_idx}/{self.data_len}] ({progress:.1f}%) | "
            f"Loss: {self.loss_meter.avg:.4f} | "
            f"LR: {lr:.2e} | "
            f"Throughput: {self.throughput_meter.avg:.0f} samples/s | "
            f"Data: {self.data_time_meter.avg*1000:.1f}ms | "
            f"Batch: {self.batch_time_meter.avg*1000:.1f}ms"
        )
        self._log(msg)
        print(msg)
    
    def _quantize_state_dict(self, state_dict):
        """对 state_dict 中 QAT 目标层的 weight 做 INT8 量化反量化.
        
        用于保存权重时确保 evaluation 加载的权重也经过量化,
        保持训练和评估的一致性.
        
        量化目标 (与 _setup_qat hooks 一致):
          - LSTM: rnn.weight_ih_l0, rnn.weight_hh_l0
          - LinearCRFEncoder: linear.weight
          
        注意: Serial (nn.Sequential) 用数字索引命名子模块, 所以 key 形如
        'encoder.9.linear.weight', 不含 'linear_crf' 字样.
        """
        if not self.args.qat_int8:
            return state_dict
        
        quantized = {}
        for key, value in state_dict.items():
            # LSTM weights
            if 'rnn.weight_ih_l0' in key or 'rnn.weight_hh_l0' in key:
                quantized[key] = fake_quantize_int8_detached(value)
            # LinearCRFEncoder's linear.weight
            # 在 rnn_encoder 架构中, 只有 LinearCRFEncoder 有 linear.weight 结尾的 key
            elif key.endswith('linear.weight'):
                quantized[key] = fake_quantize_int8_detached(value)
            else:
                quantized[key] = value
        return quantized
    
    def _validate_only(self):
        """只执行验证，不保存模型（用于训练开始前的初始验证）
        
        与原始脚本一致：只有主进程执行验证
        """
        if not self.is_main_process:
            return
        
        weights_path = os.path.join(self.res_dir, "weights_init.tar")
        model_state = self._quantize_state_dict(self.model.module.state_dict())
        torch.save(model_state, weights_path)
        
        res = model_eval(
            dataloader=self.valid_loader,
            model_dir=self.res_dir,
            weight_path=weights_path,
            is_half=True,
            device=self.local_rank,
        )
        
        mean_acc, median_acc, duration, speed_samples, speed_bases, chunks_num = res
        self._log(f"  Mean accuracy:    {mean_acc:.2f}%")
        self._log(f"  Median accuracy:  {median_acc:.2f}%")
        self._log(f"  Validation time:  {duration:.2f}s")
        
        # 清理临时文件
        if os.path.exists(weights_path):
            os.remove(weights_path)
    
    def _validate_and_save(self):
        """验证并保存模型
        
        与原始脚本完全一致：只有主进程执行验证和保存
        其他进程直接返回，不等待
        DDP 会在下一个 epoch 的 backward 时自动同步
        """
        if not self.is_main_process:
            return
        
        # 保存模型权重 (QAT 模式下对目标权重做 INT8 量化反量化)
        model_state = self._quantize_state_dict(self.model.module.state_dict())
        weights_path = os.path.join(self.res_dir, f"weights_{self.epoch}.tar")
        torch.save(model_state, weights_path)
        
        # 执行验证
        self._log(f"[Validation] Epoch {self.epoch}")
        
        res = model_eval(
            dataloader=self.valid_loader,
            model_dir=self.res_dir,
            weight_path=weights_path,
            is_half=True,
            device=self.local_rank,
        )
        
        mean_acc, median_acc, duration, speed_samples, speed_bases, chunks_num = res
        
        self._log(f"  Mean accuracy:    {mean_acc:.2f}%")
        self._log(f"  Median accuracy:  {median_acc:.2f}%")
        self._log(f"  Validation time:  {duration:.2f}s")
        self._log(f"  Speed:            {speed_samples:.2E} samples/s, {speed_bases:.2f} bases/s")
        
        # Tensorboard
        if self.log_writer:
            self.log_writer.add_scalar("val/mean_accuracy", mean_acc, self.epoch)
            self.log_writer.add_scalar("val/median_accuracy", median_acc, self.epoch)
        
        # 检查是否是最佳模型
        is_best = mean_acc > self.best_metric
        if is_best:
            self.best_metric = mean_acc
            self._log(f"  [NEW BEST] Accuracy improved to {mean_acc:.2f}%")
        
        # 保存检查点
        state = {
            'epoch': self.epoch + 1,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'args': vars(self.args),
        }
        
        # 保存 pruner 状态，以便 resume 时正确恢复 LR scheduler
        if self.pruner is not None:
            idx = self.pruner.index
            state['pruner_state'] = {
                'step': self.pruner.step,
                'index': idx,
                'lr_scheduler_state': self.pruner.optim_schedulers[idx].state_dict(),
            }
        
        self.ckpt_manager.save(state, self.epoch, self.global_step, is_best)
        
        # 保存训练参数和评估结果
        self._save_metrics(res)
    
    def _save_metrics(self, res):
        """保存评估指标"""
        mean_acc, median_acc, duration, speed_samples, speed_bases, chunks_num = res
        
        # 训练参数
        with open(f'{self.res_dir}/training_params.json', 'w') as f:
            json.dump(vars(self.args), f, indent=4)
        
        # 评估结果
        chunk_acc = {
            'epoch': self.epoch,
            'mean': round(mean_acc, 2),
            'median': round(median_acc, 2),
            'tooktime': round(duration, 2),
            'speed(samples/sec)': round(speed_samples, 2),
            'speed(bases/sec)': round(speed_bases, 2),
            'val_chunks_num': int(chunks_num),
        }
        with open(f'{self.res_dir}/chunk_acc.json', 'w') as f:
            json.dump(chunk_acc, f, indent=4)
    
    def _final_evaluation(self):
        """最终评估
        
        与原始脚本一致：只有主进程执行
        """
        if not self.is_main_process:
            return
        
        # 加载最佳模型
        best_ckpt = self.ckpt_manager.load_best()
        best_weights_path = os.path.join(self.res_dir, "weights_best.tar")
        
        if best_ckpt:
            self.model.module.load_state_dict(best_ckpt['model_state_dict'])
            best_state = self._quantize_state_dict(best_ckpt['model_state_dict'])
            torch.save(best_state, best_weights_path)
            self._log("[Final Evaluation] Using best model")
        else:
            # 没有最佳检查点，使用最后的模型
            last_state = self._quantize_state_dict(self.model.module.state_dict())
            torch.save(last_state, best_weights_path)
            self._log("[Final Evaluation] Using last model (no best checkpoint)")
        
        res = model_eval(
            dataloader=self.valid_loader,
            model_dir=self.res_dir,
            weight_path=best_weights_path,
            is_half=True,
            device=self.local_rank,
        )
        
        self._log(f"  Final Mean accuracy:   {res[0]:.2f}%")
        self._log(f"  Final Median accuracy: {res[1]:.2f}%")


# ============================================================================
# 参数解析
# ============================================================================

def get_parser():
    parser = argparse.ArgumentParser(description='高效分布式训练脚本')
    
    # 数据参数
    parser.add_argument("--data_dir", type=str, required=True, help="训练数据目录")
    parser.add_argument("--data_type", type=str, default="index", choices=["index", "pt"],
                        help="数据类型: index (HDF5) 或 pt (预转换)")
    parser.add_argument("--tokenization", type=str, default="kmer")
    parser.add_argument("--val_size", type=int, default=20000)
    
    # 模型参数
    parser.add_argument("--model", type=str, default="lstm_ctc_crf", help="模型类型")
    parser.add_argument("--pre_trained_params_file", type=str, default="", help="预训练模型路径")
    parser.add_argument("--compile", action="store_true", help="使用 torch.compile 加速 (PyTorch 2.0+)")
    
    # 训练参数
    parser.add_argument("--epoch_num", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="单卡 batch size")
    parser.add_argument("--val_batch_size", type=int, default=16, help="验证 batch size")
    parser.add_argument("--grad_accum", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=5e-4, help="学习率")
    parser.add_argument("--warmup_steps", type=int, default=100, help="warmup 步数")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--limit_train_size", type=int, default=0, help="限制训练 batch 数 (0=不限制)")
    
    # 混合精度
    parser.add_argument("--use_amp", action="store_true", default=True, help="使用混合精度训练")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    
    # 数据加载
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader 工作进程数")
    # 默认关闭 prefetch 以保持与原始代码的数值一致性
    # 如需启用，使用 --use_prefetch 参数
    parser.add_argument("--use_prefetch", action="store_true", default=False, help="使用 CUDA 数据预取 (可能略微影响数值精度)")
    parser.add_argument("--no_prefetch", dest="use_prefetch", action="store_false")
    
    # INT8 QAT 参数
    parser.add_argument("--qat_int8", action="store_true", default=False,
                        help="启用 INT8 weight QAT (量化反量化 + STE 训练)")
    parser.add_argument("--qat_weight_path", type=str, default="",
                        help="QAT 模式下加载的预训练权重路径 (e.g., weights_40.tar)")
    
    # 剪枝参数
    parser.add_argument("--pruning", type=int, default=1, help="是否启用剪枝 (0/1)")
    parser.add_argument("--sparsity", type=float, default=0.833333, help="目标稀疏度")
    parser.add_argument("--last_linear_sparsity", type=float, default=None,
                        help="最后一个 linear 层的稀疏率 (例如 0.75=4倍, 0=不剪枝, 默认跟随全局 sparsity)")
    parser.add_argument("--prune_log", type=str, default="./prune", help="剪枝日志路径")
    
    # 日志和检查点
    parser.add_argument("--output_name", type=str, required=True, help="输出目录名")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--resume", action="store_true", help="从最新检查点恢复训练")
    parser.add_argument("--val_before_train", action="store_true", help="训练开始前执行一次验证")
    
    # 其他
    parser.add_argument("--seed", type=int, default=25, help="随机种子")
    parser.add_argument("--local_rank", type=int, default=-1, help="DDP local rank (自动设置)")
    parser.add_argument("--nondeterministic", action="store_true", default=False)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--gpu_nums", type=int, default=8)
    parser.add_argument("--part_num", type=int, default=0)
    
    return parser


# ============================================================================
# 主函数
# ============================================================================

def main():
    # 设置环境变量
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '23333'
    
    args = get_parser().parse_args()
    
    # 创建训练器并开始训练
    trainer = Trainer(args)
    trainer.setup()
    trainer.train()


if __name__ == "__main__":
    main()
