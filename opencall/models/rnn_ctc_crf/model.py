"""
opencall CTC-CRF Model.
"""

from numpy.core.arrayprint import dtype_is_implied
import torch
import numpy as np
from opencall.models.common.nn import (
    Module,
    Convolution,
    LinearCRFEncoder,
    Serial,
    Permute,
    layers,
    from_dict,
)
# import sys
# sys.path.append('/store/zjj/coding/tmp/opencall/opencall/libs')

from seqdist import sparse
from seqdist.ctc_simple import logZ_cupy, viterbi_alignments
from seqdist.core import SequenceDist, Max, Log, semiring

try:
    from koi.decode import beam_search as koi_beam_search, to_str
    KOI_AVAILABLE = True
except ImportError:
    KOI_AVAILABLE = False

import time


def manual_logsumexp(x, dim=-1, keepdim=False):
    """
    手动实现 logsumexp，数值稳定版本
    
    等价于: torch.logsumexp(x, dim=dim, keepdim=keepdim)
    """
    # 步骤1: 找到最大值（防止 exp 溢出）
    x_max = x.max(dim=dim, keepdim=True)[0]
    
    # 步骤2: 减去最大值
    x_shifted = x - x_max
    
    # 步骤3: exp -> sum -> log
    result = x_max + torch.log(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True))
    
    # 步骤4: 处理 keepdim
    if not keepdim:
        result = result.squeeze(dim)
    
    return result

def get_stride(m):
    if hasattr(m, "stride"):
        return m.stride if isinstance(m.stride, int) else m.stride[0]
    if isinstance(m, Convolution):
        return get_stride(m.conv)
    if isinstance(m, Serial):
        return int(np.prod([get_stride(x) for x in m]))
    return 1

class CTC_CRF(SequenceDist):
    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat(
            [
                torch.arange(self.n_base ** (self.state_len))[:, None],
                torch.arange(self.n_base ** (self.state_len))
                .repeat_interleave(self.n_base)
                .reshape(self.n_base, -1)
                .T,
            ],
            dim=1,
        ).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base ** (self.state_len)

    def save_viterbi_inputs_with_precomputed(self, scores, out_paths, save_path="viterbi_inputs.pth"):
        """
        保存 viterbi_fused_guided_fast 函数的所有输入，包括预计算的固定索引
        """
        n_alphabet = len(self.alphabet)
        idx = self.idx
        
        # 预计算转置索引
        idx_T = idx.flatten().argsort().reshape(*idx.shape)  # (C, NZ)
        idx_T_targets = idx_T // n_alphabet                   # (C, NZ) - 目标状态
        
        data = {
            # 网络输入
            'scores': scores.cpu(),

            # 超参数
            'n_base': self.n_base,
            'state_len': self.state_len,
            'alphabet': self.alphabet,
            
            # 固定索引
            'idx': idx.cpu(),
            'idx_T': idx_T.cpu(),
            'idx_T_targets': idx_T_targets.cpu(),

            # 算法输出
            'path': out_paths.cpu(),
        }
        torch.save(data, save_path)
        print(f"Saved viterbi inputs to {save_path}")

    def logZ(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.logZ(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return scores - self.logZ(scores)[:, None] / len(scores)

    def forward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.fwd_scores_cupy(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring = Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base ** (self.state_len)), S.one)
        return sparse.bwd_scores_cupy_python(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        # add bwd scores to edge scores
        log_trans_probs = (
            scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None]
        )
        # transpose from (new_state, dropped_base) to (old_state, emitted_base) layout
        log_trans_probs = torch.cat(
            [
                log_trans_probs[:, :, :, [0]],
                log_trans_probs[:, :, :, 1:]
                .transpose(3, 2)
                .reshape(T, N, -1, self.n_base),
            ],
            dim=-1,
        )
        # convert from log probs to probs by exponentiating and normalising
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        # convert first bwd score to initial state probabilities
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = (
            T,
            N,
            *(self.n_base for _ in range(self.state_len)),
            self.n_base + 1,
        )
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(
            scores[..., 0]
            .permute(0, 1, *range(self.state_len + 1, 1, -1))
            .reshape(T, N, -1, 1),
            [0, 2],
        )
        emissions = torch.flip(
            scores[..., 1:]
            .permute(
                0,
                1,
                *range(self.state_len, 1, -1),
                self.state_len + 2,
                self.state_len + 1
            )
            .reshape(T, N, -1, self.n_base),
            [0, 2, 3],
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths


    def viterbi_posteriors(self, scores):
        """
        基于 posteriors 的 Viterbi 解码（完全展开版本）
        
        实现原始注释代码的逻辑：
        post_scores = self.posteriors(x, Log) + 1e-8
        paths = self.viterbi(post_scores.log())
        
        完全展开：
        - posteriors(x, S) = grad(logZ(x, S)) 
          = S.dsum(S.mul(alphas, betas))
        - viterbi 就是 posteriors(x, Max).argmax
        """
        T, N, _ = scores.shape
        n_states = self.n_base ** self.state_len
        n_alphabet = len(self.alphabet)
        device = scores.device
        idx = self.idx.to(device=device, dtype=torch.long)
        
        scores_float = scores.to(torch.float32)
        Ms = scores_float.reshape(T, N, n_states, n_alphabet)
        
        # 构造转置索引
        if not hasattr(self, '_idx_T') or self._idx_T.device != device:
            idx_T = idx.flatten().argsort().reshape(*idx.shape).to(device)
            idx_T_targets = idx_T // n_alphabet
            self._idx_T = idx_T
            self._idx_T_targets = idx_T_targets
        
        idx_T = self._idx_T
        idx_T_targets = self._idx_T_targets
        Ms_flat = Ms.reshape(T, N, -1)
        Ms_T = Ms_flat[:, :, idx_T]
        
        # ========================================
        # 步骤1: posteriors(scores, Log)
        # 根据 sparse.py backward 实现：
        #   alphas = forward_scores(Ms, Log)  # 已在 forward 中计算
        #   betas = backward_scores(Ms, Log)
        #   grad = Log.mul(alphas, betas[1:])  # Log.mul = add
        #   grad = Log.dsum(grad, dim=-1)      # Log.dsum = softmax
        # ========================================
        
        # Forward pass (Log semiring: logsumexp)
        # 提前分配内存，避免 list append + stack
        alphas_all = torch.zeros(T + 1, N, n_states, device=device, dtype=torch.float32)
        alpha = alphas_all[0]  # 初始状态全为0
        
        for t in range(T):
            alpha_indexed = alpha[:, idx]
            candidates = alpha_indexed + Ms[t]  # Log.mul = add
            alpha = torch.logsumexp(candidates, dim=-1)  # Log.sum = logsumexp
            alphas_all[t + 1] = alpha
        
        # Backward pass (Log semiring: logsumexp)
        # 提前分配内存，直接在正确位置写入（避免反转）
        betas_all = torch.zeros(T + 1, N, n_states, device=device, dtype=torch.float32)
        beta = betas_all[T]  # 终止状态全为0
        
        for t in range(T - 1, -1, -1):
            beta_indexed = beta[:, idx_T_targets]
            candidates = Ms_T[t] + beta_indexed  # Log.mul = add
            beta = torch.logsumexp(candidates, dim=-1)  # Log.sum = logsumexp
            betas_all[t] = beta
        
        # 计算 posteriors (边缘概率)
        # grad = alphas[t, idx[c,z]] + Ms[t,c,z] + betas[t+1,c]
        edge_grad = torch.zeros(T, N, n_states, n_alphabet, device=device, dtype=torch.float32)
        for t in range(T):
            alpha_indexed = alphas_all[t][:, idx]
            edge_grad[t] = alpha_indexed + Ms[t] + betas_all[t + 1][:, :, None]  # Log.mul = add
        
        # Log.dsum = softmax (归一化)
        post_scores_reshaped = torch.softmax(edge_grad.reshape(T, N, -1), dim=2)  # Log.dsum
        post_scores = post_scores_reshaped.reshape(T, N, n_states * n_alphabet) + 1e-8
        
        # ========================================
        # 步骤2: viterbi(log(posteriors))
        # Viterbi 只需要 Max semiring 的 forward pass + traceback
        # 不需要 backward！
        # ========================================
        
        log_post = torch.log(post_scores)
        log_Ms = log_post.reshape(T, N, n_states, n_alphabet)
        
        # Viterbi forward pass (Max semiring: max)
        alpha_max = torch.full((N, n_states), float('-inf'), device=device, dtype=torch.float32)
        alpha_max[:, 0] = 0.0
        traceback = torch.zeros(T, N, n_states, dtype=torch.int8, device=device)
        
        for t in range(T):
            alpha_indexed = alpha_max[:, idx]  # (N, n_states, n_alphabet)
            candidates = alpha_indexed + log_Ms[t]  # Max.mul = add
            alpha_max, best_z = candidates.max(dim=-1)  # Max.sum = max，同时记录 argmax
            traceback[t] = best_z.to(torch.int8)
        
        # 回溯得到最优路径
        current_states = alpha_max.argmax(dim=-1)  # 找到最优终止状态
        paths = torch.zeros(T, N, dtype=torch.int8, device=device)
        batch_idx = torch.arange(N, device=device)
        
        for t in range(T - 1, -1, -1):
            best_edges = traceback[t, batch_idx, current_states]
            paths[t] = best_edges
            current_states = idx[current_states, best_edges.long()]
        
        return paths.T.to(torch.long)

    def viterbi_guided_bidirectional(self, scores, use_bfloat16=True):
        """
        双向引导 Viterbi（精度和速度的平衡，支持 bfloat16）
        
        性能优化：
        - 保留完整的 forward + backward 信息（高精度）
        - 跳过 softmax 归一化（速度提升 20-30%）
        - 直接在 log 空间做 Viterbi
        - 支持 bfloat16 计算（内存和速度优化）
        - 每10步归一化（数值稳定性）
        
        理论依据：
        alpha + Ms + beta 已经包含完整的双向信息，
        Viterbi 只需要相对分数，不需要归一化为概率分布
        
        介于 viterbi_posteriors 和 viterbi_fused_guided_fast 之间：
        - 比 posteriors 快（省略 softmax + log）
        - 比 fused_fast 准确（包含 forward 信息）
        
        Args:
            scores: 输入分数 (T, N, C)
            use_bfloat16: 是否使用 bfloat16 计算（默认 False，使用 float32）

        具体参数值：
            - T = 1000
            - N = 512 (batch size)
            - n_states = 1024
            - self.state_len = 5 
            - self.n_base = 4


        """
        T, N, _ = scores.shape
        n_states = self.n_base ** self.state_len
        n_alphabet = len(self.alphabet)
        device = scores.device
        idx = self.idx.to(device=device, dtype=torch.long)
        
        # 选择数据类型
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        Ms = scores.to(dtype).reshape(T, N, n_states, n_alphabet)
        
        # 构造转置索引
        if not hasattr(self, '_idx_T') or self._idx_T.device != device:
            idx_T = idx.flatten().argsort().reshape(*idx.shape).to(device)
            idx_T_targets = idx_T // n_alphabet
            self._idx_T = idx_T
            self._idx_T_targets = idx_T_targets
        
        idx_T = self._idx_T
        idx_T_targets = self._idx_T_targets
        Ms_flat = Ms.reshape(T, N, -1)
        Ms_T = Ms_flat[:, :, idx_T]
        
        # 归一化间隔（bfloat16 需要更频繁的归一化）
        # 更小的 segment_size = 更频繁的归一化 = 更好的数值稳定性
        segment_size = 10
        
        # ========================================
        # 步骤1: Log Forward (manual_logsumexp + 归一化)
        # ========================================
        alphas_all = torch.zeros(T + 1, N, n_states, device=device, dtype=dtype)
        alpha = alphas_all[0]  # 初始状态全为0
        
        for t in range(T):
            alpha_indexed = alpha[:, idx]
            candidates = alpha_indexed + Ms[t]
            alpha = torch.logsumexp(candidates, dim=-1)
            
            # 每 segment_size 步归一化（保持数值稳定）
            if t % segment_size == 0:
                alpha_min = alpha.min(dim=1, keepdim=True)[0]
                alpha = alpha - alpha_min
            
            alphas_all[t + 1] = alpha
        
        # ========================================
        # 步骤2: Log Backward (manual_logsumexp + 归一化)
        # ========================================
        betas_all = torch.zeros(T + 1, N, n_states, device=device, dtype=dtype)
        beta = betas_all[T]  # 终止状态全为0
        
        for t in range(T - 1, -1, -1):
            beta_indexed = beta[:, idx_T_targets]
            candidates = Ms_T[t] + beta_indexed
            beta = torch.logsumexp(candidates, dim=-1)
            
            # 每 segment_size 步归一化（保持数值稳定）
            if t % segment_size == 0:
                beta_min = beta.min(dim=1, keepdim=True)[0]
                beta = beta - beta_min
            
            betas_all[t] = beta
        
        # ========================================
        # 步骤3 + 4: 融合计算引导分数并做 Viterbi forward
        # 避免额外的内存分配，直接在线计算
        # ========================================
        alpha_max = torch.full((N, n_states), float('-inf'), device=device, dtype=dtype)
        alpha_max[:, 0] = 0.0
        traceback = torch.zeros(T, N, n_states, dtype=torch.int8, device=device)
        
        for t in range(T):
            # 在线计算 guided_scores = alpha[src] + Ms + beta[dst]
            alpha_indexed_src = alphas_all[t][:, idx]
            beta_indexed_dst = betas_all[t + 1][:, :, None]
            guided_scores_t = alpha_indexed_src + Ms[t] + beta_indexed_dst
            
            # Viterbi forward: alpha_max[dst] = max(alpha_max[src] + guided_scores)
            alpha_indexed_max = alpha_max[:, idx]
            candidates = alpha_indexed_max + guided_scores_t
            alpha_max, best_z = candidates.max(dim=-1)  # 用 float 保证精度
            traceback[t] = best_z.to(torch.int8)

            if t % 5 == 0:
                alpha_max = alpha_max-alpha_max.max(dim=-1, keepdim=True)[0]

        # ========================================
        # 步骤5: 回溯得到最优路径 (CPU)
        # ========================================
        current_states = alpha_max.argmax(dim=-1)
        paths = torch.zeros(T, N, dtype=torch.int8, device=device)
        batch_idx = torch.arange(N, device=device)
        
        for t in range(T - 1, -1, -1):
            best_edges = traceback[t, batch_idx, current_states]
            paths[t] = best_edges
            current_states = idx[current_states, best_edges.long()]
        
        return paths.T.to(torch.long)

    def viterbi_fused_guided_fast(self, scores):
        """
        快速版带引导的 Viterbi（优化实现）
        
        使用转置索引正确计算后向分数：
        - idx[c, z] 表示到达状态 c 的第 z 条入边的来源状态
        - 后向计算需要从"出边"视角，因此需要对 idx 和 Ms 做转置
        """
        T, N, _ = scores.shape
        n_states = self.n_base ** self.state_len
        n_alphabet = len(self.alphabet)
        
        Ms = scores.reshape(T, N, n_states, n_alphabet)
        device = scores.device
        dtype = scores.dtype
        idx = self.idx.to(device=device, dtype=torch.long)
        
        # ===== 构造转置索引（使用缓存）=====
        # idx_T: 将 idx 按来源状态分组，idx_T[source, j] 是从 source 出发的第 j 条边在原始展平 idx 中的位置
        # idx_T_targets: idx_T_targets[source, j] 是从 source 出发的第 j 条边到达的目标状态
        if not hasattr(self, '_idx_T') or self._idx_T.device != device:
            idx_T = idx.flatten().argsort().reshape(*idx.shape).to(device)  # (C, NZ)
            idx_T_targets = idx_T // n_alphabet  # (C, NZ) - 目标状态
            self._idx_T = idx_T
            self._idx_T_targets = idx_T_targets
        
        idx_T = self._idx_T
        idx_T_targets = self._idx_T_targets
        
        # ===== 后向遍历 =====
        Ms_flat = Ms.reshape(T, N, -1)  # (T, N, C*NZ)
        Ms_T = Ms_flat[:, :, idx_T]  # 转置后的 Ms: (T, N, C, NZ)，Ms_T[t, n, s, j] 是从状态 s 出发的第 j 条边的分数

        segment_size = 10
        betas_all = torch.zeros(T + 1, N, n_states, dtype=torch.bfloat16, device=device)
        beta_next = torch.zeros(N, n_states, dtype=torch.bfloat16, device=device)

        for t in range(T - 1, -1, -1):
            candidates = Ms_T[t] + beta_next[:, idx_T_targets]
            beta_next = manual_logsumexp(candidates, dim=-1)
            
            # 每10步归一化：减去最小值
            if t % segment_size == 0:
                beta_min = beta_next.min(dim=1, keepdim=True)[0]
                beta_next = beta_next - beta_min  # 核心：保持相对关系
            
            betas_all[t] = beta_next
        
        guided_Ms = Ms + betas_all[1:, :, :, None]
        # guided_Ms = guided_Ms /50
        
        # device = "cpu"
        # guided_Ms = guided_Ms.to(device)
        # idx = idx.to(device)


        #t0 = time.time()
        alpha = torch.zeros(N, n_states, device=device, dtype=dtype)
        traceback = torch.zeros(T, N, n_states, dtype=torch.int8, device=device)

        
        for t in range(T):
            candidates = alpha[:, idx] + guided_Ms[t]
            alpha, best_z = candidates.max(dim=-1)
            traceback[t] = best_z.to(torch.int8)

            if t % 5 == 0:
                alpha = alpha-alpha.max(dim=-1, keepdim=True)[0]


       # t1 = time.time()
        # print(f"First Loop Time taken: {t1 - t0} seconds")
        
        # ===== 回溯 =====
        current_states = alpha.argmax(dim=-1)
        paths = torch.zeros(T, N, dtype=torch.int8, device=device)
        batch_idx = torch.arange(N, device=device)
        
        for t in range(T - 1, -1, -1):
            best_edges = traceback[t, batch_idx, current_states]
            paths[t] = best_edges
            current_states = idx[current_states, best_edges.long()]
        #t2 = time.time()
        #print(f"Second LoopTime taken: {t2 - t1} seconds")
       # print(f"Total Time taken: {t2 - t0} seconds")

        return paths.T.to(torch.long)

    def path_to_str(self, path):
        alphabet = np.frombuffer("".join(self.alphabet).encode(), dtype="u1")
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        # convert from CTC targets (with blank=0) to zero indexed
        targets = torch.clamp(targets - 1, 0)

        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i : n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)  # indices 用于在self.idx.flatten()里找kmer的下标
        move_indices = stay_indices[:, 1:] + targets[:, : n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(
        self,
        scores,
        targets,
        target_lengths,
        loss_clip=None,
        reduction="mean",
        normalise_scores=True,
    ):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cupy(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = -(logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == "mean":
            return loss.mean()
        elif reduction in ("none", None):
            return loss
        else:
            raise ValueError("Unknown reduction type {}".format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(
            stay_scores, move_scores, target_lengths + 1 - self.state_len
        )


def conv(c_in, c_out, ks, stride=1, bias=False, activation=None):
    return Convolution(
        c_in,
        c_out,
        ks,
        stride=stride,
        padding=ks // 2,
        bias=bias,
        activation=activation,
    )


def rnn_encoder(
    n_base,
    state_len,
    insize=1,
    stride=5,
    winlen=19,
    activation="swish",
    rnn_type="lstm",
    features=768,
    scale=5.0,
    dropout=0.0,
    blank_score=None,
    expand_blanks=True,
    num_layers=5,
    bidirectional=False,
):
    """
    RNN encoder with optional bidirectional LSTM.
    
    Args:
        bidirectional: If True, use BiLSTM. Note that each BiLSTM layer will use
                      features//2 hidden size, so output remains features-dimensional.
    """
    if bidirectional and rnn_type == "lstm":
        rnn = layers["bilstm"]
        # For BiLSTM, use half hidden size so output is features-dimensional
        hidden_size = features // 2
        sublayers = [
            conv(insize, 4, ks=5, bias=True, activation=activation),
            conv(4, 16, ks=5, bias=True, activation=activation),
            conv(
                16, features, ks=winlen, stride=stride, bias=True, activation=activation
            ),
            Permute([2, 0, 1]),
        ]
        # First BiLSTM layer takes features as input, outputs 2*hidden_size=features
        # Note: In this codebase, LSTM(size, insize) creates lstm with hidden_size=insize, input_size=size
        # So BiLSTM(size=features, insize=hidden_size) gives hidden_size=hidden_size, input_size=features
        sublayers.append(rnn(features, hidden_size, dropout=dropout))
        # Subsequent layers take features (2*hidden_size) as input
        for i in range(1, num_layers):
            sublayers.append(rnn(features, hidden_size, dropout=dropout))
        
        sublayers.append(
            LinearCRFEncoder(
                features,
                n_base,
                state_len,
                activation="tanh",
                scale=scale,
                blank_score=blank_score,
                expand_blanks=expand_blanks,
            )
        )
        return Serial(sublayers)
    else:
        # Original unidirectional LSTM
        rnn = layers[rnn_type]
        return Serial(
            [
                conv(insize, 4, ks=5, bias=True, activation=activation),
                conv(4, 16, ks=5, bias=True, activation=activation),
                conv(
                    16, features, ks=winlen, stride=stride, bias=True, activation=activation
                ),
                Permute([2, 0, 1]),
                *(
                    rnn(features, features, reverse=(num_layers - i) % 2, dropout=dropout)
                    for i in range(num_layers)
                ),
                LinearCRFEncoder(
                    features,
                    n_base,
                    state_len,
                    activation="tanh",
                    scale=scale,
                    blank_score=blank_score,
                    expand_blanks=expand_blanks,
                ),
            ]
        )


class seqdistModel(Module):
    def __init__(self, encoder, seqdist):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

    def forward(self, x):
        return self.encoder(x)

    def decode_batch(self, x, beam_width=10, beam_cut=100, scale=1.0, offset=0.0, blank_score=2.0, use_koi=True, viterbi_method='bidirectional', use_bfloat16=True):
        """
        Decode a batch of scores using either koi beam_search or viterbi decoding.
        
        Args:
            x: scores tensor of shape (T, N, C)
            beam_width: beam width for beam search
            beam_cut: beam cut threshold for beam search
            scale: scale factor for beam search
            offset: offset for beam search
            blank_score: blank score for beam search
            use_koi: whether to use koi beam_search (if available)
            viterbi_method: which viterbi method to use when use_koi=False
                'posteriors' - Full posteriors + viterbi (slowest, highest accuracy)
                'bidirectional' - Bidirectional guided viterbi (balanced, recommended)
                'fast' - Fused guided fast (fastest, good accuracy)
            use_bfloat16: use bfloat16 for computation (only for 'bidirectional' and 'fast')
            
        Returns:
            list of decoded sequences
        """
        if use_koi and KOI_AVAILABLE:
            # Use koi beam_search
            # Permute from (T, N, C) to (N, T, C)
            T, N, C = x.shape
            scores = x.permute(1, 0, 2).contiguous()
            
            # Reshape to remove blank: (N, T, n_states, n_alphabet) -> (N, T, n_states, n_bases)
            # C = n_states * n_alphabet, where n_alphabet = n_base + 1 (including blank)
            n_states = self.seqdist.n_base ** self.seqdist.state_len
            n_alphabet = len(self.seqdist.alphabet)  # n_base + 1 (including blank)
            
            # Reshape: (N, T, C) -> (N, T, n_states, n_alphabet)
            scores = scores.reshape(N, T, n_states, n_alphabet)
            # Remove blank (first element): keep [:, :, :, 1:]
            scores = scores[:, :, :, 1:]
            # Reshape back: (N, T, n_states, n_base) -> (N, T, n_states * n_base)
            scores = scores.reshape(N, T, -1).contiguous()
            
            # Convert to fp16 (required by koi)
            if scores.dtype != torch.float16:
                scores = scores.to(torch.float16)
            
            # Call koi beam_search
            sequence, qstring, moves = koi_beam_search(
                scores,
                beam_width=beam_width,
                beam_cut=beam_cut,
                scale=scale,
                offset=offset,
                blank_score=blank_score,
            )
            
            # Convert each sequence to string using to_str
            results = []
            for i in range(N):
                seq_str = to_str(sequence[i])
                results.append(seq_str)
            
            return results
        else:           # Fallback to viterbi decoding
            # 选择 Viterbi 方法（性能 vs 精度权衡）
            with torch.no_grad():
                if viterbi_method == 'posteriors':
                    # 方法1: 完整 posteriors + viterbi
                    # - 最高精度（完整边缘概率 + softmax 归一化）
                    # - 最慢（3次遍历 + softmax + log）
                    # - 适用于：需要与原始 seqdist 完全一致的结果
                    paths = self.seqdist.viterbi_posteriors(x.to(torch.float32))
                    
                elif viterbi_method == 'bidirectional':
                    # 方法2: 双向引导 viterbi（推荐，默认）
                    # - 高精度（完整双向信息，无 softmax）
                    # - 中等速度（3次遍历，比 posteriors 快 20-30%）
                    # - 支持 bfloat16（内存优化）
                    # - 适用于：生产环境，平衡精度和速度
                    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                    paths = self.seqdist.viterbi_guided_bidirectional(x.to(dtype), use_bfloat16=use_bfloat16)
                    
                elif viterbi_method == 'fast':
                    # 方法3: 快速融合 viterbi
                    # - 良好精度（只用后向信息）
                    # - 最快（2次遍历，使用 bfloat16）
                    # - 适用于：实时推理，内存受限场景
                    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
                    paths = self.seqdist.viterbi_fused_guided_fast(x.to(dtype))
                    
                else:
                    raise ValueError(f"Unknown viterbi_method: {viterbi_method}. "
                                   f"Choose from 'posteriors', 'bidirectional', or 'fast'")
                    
            return [self.seqdist.path_to_str(path) for path in paths.cpu().numpy()]


    def get_path(self, x):
        with torch.no_grad():
            paths = self.seqdist.viterbi_fused(x.to(torch.float32))
        return paths.cpu().numpy()[0]

    def decode(self, x, beam_width=5, beam_cut=1e-3, scale=1.0, offset=0.0, blank_score=2.0, use_koi=True, viterbi_method='bidirectional', use_bfloat16=False):
        """
        Decode a single sample.
        
        Args:
            x: scores tensor of shape (T, C) or (T, 1, C)
            beam_width: beam width for beam search
            beam_cut: beam cut threshold for beam search
            scale: scale factor for beam search
            offset: offset for beam search
            blank_score: blank score for beam search
            use_koi: whether to use koi beam_search (if available)
            viterbi_method: which viterbi method to use ('posteriors', 'bidirectional', 'fast')
            use_bfloat16: use bfloat16 for computation (only for 'bidirectional' and 'fast')
            
        Returns:
            decoded sequence string
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (T, C) -> (T, 1, C)
        return self.decode_batch(
            x, 
            beam_width=beam_width, 
            beam_cut=beam_cut, 
            scale=scale,
            offset=offset,
            blank_score=blank_score,
            use_koi=use_koi,
            viterbi_method=viterbi_method,
            use_bfloat16=use_bfloat16
        )[0]

    def greedy_decode(self, x):
        """
        Greedy decoding guided by backward scores (equivalent to beam_search with k=1).
        
        This uses forward-backward algorithm:
        1. Compute backward scores for guidance
        2. At each time step, select transition with highest (edge_score + backward_score)
        3. Follow the greedy path through the lattice
        
        Args:
            x: scores tensor of shape (T, C) or (T, N, C)
            
        Returns:
            decoded sequence string (if T, C) or list of strings (if T, N, C)
        """
        # Handle single sample or batch
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (T, C) -> (T, 1, C)
            single_sample = True
        else:
            single_sample = False
        
        T, N, C = x.shape
        x = x.to(torch.float32)
        
        # Get model parameters
        n_states = self.seqdist.n_base ** self.seqdist.state_len
        n_alphabet = len(self.seqdist.alphabet)  # n_base + 1 (including blank)
        
        # Reshape scores: (T, N, C) -> (T, N, n_states, n_alphabet)
        scores = x.reshape(T, N, n_states, n_alphabet)
        
        # Compute backward scores for guidance: (T+1, N, n_states)
        betas = self.seqdist.backward_scores(x)
        
        # Add backward scores to edge scores
        # For transition (old_state, action) -> new_state at time t:
        # guided_score = edge_score[t, old_state, action] + beta[t+1, new_state]
        idx_long = self.seqdist.idx.long().to(betas.device)  # (n_states, n_alphabet)
        betas_for_transitions = torch.gather(
            betas[1:].unsqueeze(2).expand(T, N, n_states, n_states),  # (T, N, n_states, n_states)
            dim=3,
            index=idx_long.unsqueeze(0).unsqueeze(0).expand(T, N, -1, -1)  # (T, N, n_states, n_alphabet)
        )  # Result: (T, N, n_states, n_alphabet)
        guided_scores = scores + betas_for_transitions
        
        # Initialize: all samples start from state 0 (e.g., "AAAAA" for 5-mer)
        current_states = torch.zeros(N, dtype=torch.long, device=x.device)
        paths = []
        
        # Greedy forward pass using guided scores
        for t in range(T):
            batch_idx = torch.arange(N, device=x.device, dtype=torch.long)
            state_scores = guided_scores[t, batch_idx, current_states, :]  # (N, n_alphabet)
            
            # Greedy: select action with highest guided score
            best_actions = state_scores.argmax(dim=1)  # (N,)
            
            # Transition to next states
            next_states = self.seqdist.idx[current_states, best_actions].long()  # (N,)
            
            paths.append(best_actions.cpu())
            current_states = next_states
        
        # Convert to sequences
        paths = torch.stack(paths, dim=0).T.numpy()  # (N, T)
        results = [self.seqdist.path_to_str(paths[n]) for n in range(N)]
        
        return results[0] if single_sample else results
    
    def greedy_decode_batch(self, x):
        """
        Batch version of greedy decode.
        
        Args:
            x: scores tensor of shape (T, N, C)
            
        Returns:
            list of decoded sequences
        """
        return self.greedy_decode(x)

    def loss(self, scores, targets, target_lengths, **kwargs):
        return self.seqdist.ctc_loss(
            scores.to(torch.float32), targets, target_lengths, **kwargs
        )


class Model(seqdistModel):
    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config["global_norm"]["state_len"],
            alphabet=config["labels"]["labels"],
        )
        if "type" in config["encoder"]:  # new-style config
            encoder = from_dict(config["encoder"])
        else:  # old-style
            encoder = rnn_encoder(
                seqdist.n_base,
                seqdist.state_len,
                insize=config["input"]["features"],
                **config["encoder"]
            )
        super().__init__(encoder, seqdist)
        self.config = config
        self.tokenization = "kmer"
