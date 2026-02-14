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
        self._setup_optimizer()
        self._setup_checkpoint()
        self._try_resume()  # 必须在 pruner 之前恢复模型/优化器状态
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
        # 与原始代码保持一致: 使用默认的 AdamW 参数
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.args.lr
        )
        
        # 注意：使用梯度累积时，lr_scheduler.step() 只在参数更新时调用
        # 所以 total_steps 需要除以 grad_accum
        effective_steps_per_epoch = self.data_len // self.args.grad_accum
        total_steps = effective_steps_per_epoch * self.args.epoch_num
        
        if self.is_main_process:
            print(f"[LR Scheduler] Effective steps per epoch: {effective_steps_per_epoch}, Total steps: {total_steps}")
        
        self.lr_scheduler = get_lr_scheduler(
            epochs=total_steps,
            optimizer=self.optimizer,
            data_len=1,
            last_epoch=0,
            end_ratio=0.01,
            warmup_steps=self.args.warmup_steps,
        )
        
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
            "encoder.9.linear.weight": 0,
        }
        
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
        
        # Resume 时将 global_step 同步给 pruner
        # PrunerScheduler.step 和 Trainer.global_step 是 1:1 递增的，
        # 但 pruner 没有独立的 save/load，所以利用已保存的 global_step 恢复
        if self.args.resume and self.global_step > 0 and pruner_resume_dict is None:
            self.pruner.step = self.global_step
            self.pruner.resume_tag = True
            self.pruner.init_pruner()
            if self.is_main_process:
                print(f"[Resume] Synced pruner step to global_step={self.global_step}")
    
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
        
        # 恢复学习率调度器状态
        # 方法：重新创建 scheduler 并设置正确的 start_step
        # 这比循环调用 step() 更高效，且与原始实现一致
        effective_steps_per_epoch = self.data_len // self.args.grad_accum
        total_steps = effective_steps_per_epoch * self.args.epoch_num
        
        # 重新创建 scheduler，从正确的 step 开始
        from opencall.models.common.schedule import func_scheduler, cosine_decay_schedule, piecewise_schedule, linear_schedule
        
        func = cosine_decay_schedule(1.0, 0.01)  # end_ratio=0.01
        warmup_steps = self.args.warmup_steps
        if warmup_steps:
            y0 = func(0.0)
            func = piecewise_schedule(
                [warmup_steps / total_steps], 
                [linear_schedule(0.1 * y0, y0), func]  # warmup_ratio=0.1
            )
        
        from torch.optim.lr_scheduler import LambdaLR
        # 注意：必须在 lambda 外部捕获 start_step，否则会引用变化的 self.global_step
        start_step = self.global_step
        self.lr_scheduler = LambdaLR(
            self.optimizer, 
            lambda step, s=start_step: func((step + s) / total_steps)
        )
        
        if self.is_main_process:
            print(f"[Resume] Loaded checkpoint from epoch {self.epoch}, step {self.global_step}")
            print(f"[Resume] LR scheduler recreated with start_step={self.global_step}, current LR: {self.lr_scheduler.get_last_lr()[0]:.2e}")
    
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
                    'lr': f'{self.lr_scheduler.get_last_lr()[0]:.2e}',
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
            
            # 学习率更新 (与原始代码一致: 每个 batch 后更新)
            self.lr_scheduler.step()
            
            # 剪枝
            if self.pruner is not None:
                self.pruner.prune()
            
            self.global_step += 1
            
            # Tensorboard
            if self.is_main_process and self.log_writer and self.global_step % 100 == 0:
                self.log_writer.add_scalar("train/loss", losses["loss"].item(), self.global_step)
                self.log_writer.add_scalar("train/lr", self.lr_scheduler.get_last_lr()[0], self.global_step)
                self.log_writer.add_scalar("train/throughput", self.throughput_meter.avg, self.global_step)
        
        return losses["loss"].item()
    
    def _log_step(self, batch_idx: int):
        """记录训练步骤日志"""
        progress = batch_idx / self.data_len * 100
        lr = self.lr_scheduler.get_last_lr()[0]
        
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
    
    def _validate_only(self):
        """只执行验证，不保存模型（用于训练开始前的初始验证）
        
        与原始脚本一致：只有主进程执行验证
        """
        if not self.is_main_process:
            return
        
        weights_path = os.path.join(self.res_dir, "weights_init.tar")
        torch.save(self.model.module.state_dict(), weights_path)
        
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
        
        # 保存模型权重
        model_state = self.model.module.state_dict()
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
            torch.save(best_ckpt['model_state_dict'], best_weights_path)
            self._log("[Final Evaluation] Using best model")
        else:
            # 没有最佳检查点，使用最后的模型
            torch.save(self.model.module.state_dict(), best_weights_path)
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
    parser.add_argument("--warmup_steps", type=int, default=200, help="warmup 步数 (与原脚本一致)")
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
    
    # 剪枝参数
    parser.add_argument("--pruning", type=int, default=1, help="是否启用剪枝 (0/1)")
    parser.add_argument("--sparsity", type=float, default=0.833333, help="目标稀疏度")
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
