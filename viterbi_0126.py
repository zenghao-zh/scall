import torch 
import numpy as np

import torch.onnx
import torch.nn.functional as F

class SimpleMatmul(torch.nn.Module):
    def __init__(self, K=5120, N=5120):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(N, K))  # 注意：推荐存 [N,K]，利于后端 dense
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        # x: [1000, 256, 5120]
        x2 = x.reshape(-1, x.shape[-1])          # [-1, K]
        y2 = F.linear(x2, self.weight)           # y2 = x2 @ weight.T，输出 [-1, N]
        return y2.reshape(*x.shape[:-1], -1)

# self.n_base = 4
# self.state_len = 5
# self.alphabet = ['N', 'A', 'C', 'G', 'T']
# scores.shape: [seqlen, batch_size, num_features] -> torch.Size([1000, 256, 5120])

class Decoder:
    def __init__(self, data_path='viterbi_inputs.pth', device = 'cuda:0', dtype= torch.float32):
        data = torch.load(data_path)

        # 取出所有预计算值
        self.scores = data['scores'].to(device=device, dtype=dtype)
        self.idx = data['idx'].to(device, dtype=torch.long)
        self._idx_T = data['idx_T'].to(device)
        self._idx_T_targets = data['idx_T_targets'].to(device)
        self.n_base = data['n_base']
        self.state_len = data['state_len']
        self.alphabet = data['alphabet']
        self.true_paths = data['path'].to(device)
        self._onnx_exported = False

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


def viterbi_guided_bidirectional_reshape(self, scores, use_bfloat16=True):
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
        Ms = scores.transpose(1,2).to(dtype).reshape(T, n_states, n_alphabet, N)
        
        # 构造转置索引
        if not hasattr(self, '_idx_T') or self._idx_T.device != device:
            idx_T = idx.flatten().argsort().reshape(*idx.shape).to(device)
            idx_T_targets = idx_T // n_alphabet
            self._idx_T = idx_T
            self._idx_T_targets = idx_T_targets
        
        idx_T = self._idx_T
        idx_T_targets = self._idx_T_targets
        Ms_flat = Ms.reshape(T, -1, N)
        Ms_T = Ms_flat[:, idx_T, :]
        
        # 归一化间隔（bfloat16 需要更频繁的归一化）
        # 更小的 segment_size = 更频繁的归一化 = 更好的数值稳定性
        segment_size = 10
        
        # ========================================
        # 步骤1: Log Forward (manual_logsumexp + 归一化)
        # ========================================
        alphas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype)
        alpha = alphas_all[0]  # 初始状态全为0
        
        for t in range(T):
            alpha_indexed = alpha[idx, :]
            candidates = alpha_indexed + Ms[t]
            alpha = torch.logsumexp(candidates, dim=1)
            
            # 每 segment_size 步归一化（保持数值稳定）
            if t % segment_size == 0:
                alpha_min = alpha.min(dim=0, keepdim=True)[0]
                alpha = alpha - alpha_min
            
            alphas_all[t + 1] = alpha
        
        # ========================================
        # 步骤2: Log Backward (manual_logsumexp + 归一化)
        # ========================================
        betas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype)
        beta = betas_all[T]  # 终止状态全为0
        
        for t in range(T - 1, -1, -1):
            beta_indexed = beta[idx_T_targets, :]
            candidates = Ms_T[t] + beta_indexed
            beta = torch.logsumexp(candidates, dim=1)
            
            # 每 segment_size 步归一化（保持数值稳定）
            if t % segment_size == 0:
                beta_min = beta.min(dim=0, keepdim=True)[0]
                beta = beta - beta_min
            
            betas_all[t] = beta
        
        # ========================================
        # 步骤3 + 4: 融合计算引导分数并做 Viterbi forward
        # 避免额外的内存分配，直接在线计算
        # ========================================
        alpha_max = torch.full((n_states, N), float('-inf'), device=device, dtype=dtype)
        alpha_max[0, :] = 0.0
        traceback = torch.zeros(T, n_states, N, dtype=torch.int8, device=device)
        
        for t in range(T):
            # 在线计算 guided_scores = alpha[src] + Ms + beta[dst]
            alpha_indexed_src = alphas_all[t][idx, :]
            beta_indexed_dst = betas_all[t + 1][:, None, :]
            guided_scores_t = alpha_indexed_src + Ms[t] + beta_indexed_dst
            
            # Viterbi forward: alpha_max[dst] = max(alpha_max[src] + guided_scores)
            alpha_indexed_max = alpha_max [idx, :]
            candidates = alpha_indexed_max + guided_scores_t
            alpha_max, best_z = candidates.max(dim=1)  # 用 float 保证精度
            traceback[t] = best_z.to(torch.int8)

            if t % 5 == 0:
                alpha_max = alpha_max-alpha_max.max(dim=0, keepdim=True)[0]

        # ========================================
        # 步骤5: 回溯得到最优路径 (CPU)
        # ========================================
        current_states = alpha_max.argmax(dim=0)
        paths = torch.zeros(T, N, dtype=torch.int8, device=device)
        batch_idx = torch.arange(N, device=device)
        
        for t in range(T - 1, -1, -1):
            best_edges = traceback[t, current_states, batch_idx]
            paths[t] = best_edges
            current_states = idx[current_states, best_edges.long()]
        
        return paths.T.to(torch.long)


if __name__ == '__main__':
    decoder = Decoder(data_path='viterbi_inputs.pth', device='cuda:0')
    paths = decoder.viterbi_fused_guided_fast(decoder.scores)

    # 计算相似度（匹配率）
    similarity = (paths == decoder.true_paths).float().mean().item()
    print(f'Similarity: {similarity * 100:.2f}%')
    
    if similarity > 0.999:
        print(f'⚠️  test viterbi_fused_guided_fast passed ({similarity * 100:.2f}% match)')
    else:
        print(f'❌ test viterbi_fused_guided_fast failed ({similarity * 100:.2f}% match)')


# def viterbi_fused_guided_fast(self, scores):
#         """
#         快速版带引导的 Viterbi（优化实现）
        
#         使用转置索引正确计算后向分数：
#         - idx[c, z] 表示到达状态 c 的第 z 条入边的来源状态
#         - 后向计算需要从"出边"视角，因此需要对 idx 和 Ms 做转置
#         """
#         T, N, _ = scores.shape
#         n_states = self.n_base ** self.state_len
#         n_alphabet = len(self.alphabet)
        
#         Ms = scores.reshape(T, N, n_states, n_alphabet)
#         device = scores.device
#         dtype = scores.dtype
#         idx = self.idx.to(device=device, dtype=torch.long)
        
#         # ===== 构造转置索引（使用缓存）=====
#         # idx_T: 将 idx 按来源状态分组，idx_T[source, j] 是从 source 出发的第 j 条边在原始展平 idx 中的位置
#         # idx_T_targets: idx_T_targets[source, j] 是从 source 出发的第 j 条边到达的目标状态
#         if not hasattr(self, '_idx_T') or self._idx_T.device != device:
#             idx_T = idx.flatten().argsort().reshape(*idx.shape).to(device)  # (C, NZ)
#             idx_T_targets = idx_T // n_alphabet  # (C, NZ) - 目标状态
#             self._idx_T = idx_T
#             self._idx_T_targets = idx_T_targets
        
#         idx_T = self._idx_T
#         idx_T_targets = self._idx_T_targets
        
#         # ===== 后向遍历 =====
#         Ms_flat = Ms.reshape(T, N, -1)  # (T, N, C*NZ)
#         Ms_T = Ms_flat[:, :, idx_T]  # 转置后的 Ms: (T, N, C, NZ)，Ms_T[t, n, s, j] 是从状态 s 出发的第 j 条边的分数

#         segment_size = 10
#         betas_all = torch.zeros(T + 1, N, n_states, dtype=torch.bfloat16, device=device)
#         beta_next = torch.zeros(N, n_states, dtype=torch.bfloat16, device=device)

#         for t in range(T - 1, -1, -1):
#             candidates = Ms_T[t] + beta_next[:, idx_T_targets]
#             beta_next = torch.logsumexp(candidates, dim=-1).bfloat16()
            
#             # 每10步归一化：减去最小值
#             if t % segment_size == 0:
#                 beta_min = beta_next.min(dim=1, keepdim=True)[0]
#                 beta_next = beta_next - beta_min  # 核心：保持相对关系
            
#             betas_all[t] = beta_next
        
#         # beta_next = torch.zeros(N, n_states, device=device, dtype=torch.float32)
#         # betas_all = torch.zeros(T + 1, N, n_states, device=device, dtype=torch.float32)

#         # for t in range(T - 1, -1, -1):
#         #     # 确保 Ms_T 和 idx_T_targets 相关的张量也是 bf16
#         #     candidates = Ms_T[t] + beta_next[:, idx_T_targets]  # (N, C, NZ)
            
#         #     # 手动实现 logsumexp 以保持 bf16
#         #     # PyTorch 的 logsumexp 可能会内部转换为 fp32
#         #    #  beta_indexed = torch.nn.functional.embedding(idx_T_targets, beta_next.T).permute(2, 0, 1)
#         #    #  candidates = Ms_T[t] + beta_indexed  # (N, C, NZ)
                
#         #         # 精确 logsumexp
#         #     beta_next = torch.logsumexp(candidates.float(), dim=-1)
#         #     betas_all[t] = beta_next
        
#         # ===== 前向遍历 + 引导 =====
#         # guided_Ms[t, n, c, z] = Ms[t, n, c, z] + beta[t+1, n, c]
#         # 边分数 + 到达状态 c 后的最优剩余分数
#         guided_Ms = Ms + betas_all[1:, :, :, None]
        
#         alpha = torch.zeros(N, n_states, device=device, dtype=dtype)
#         traceback = torch.zeros(T, N, n_states, dtype=torch.int8, device=device)
        
#         for t in range(T):
#             candidates = alpha[:, idx] + guided_Ms[t]
#             alpha, best_z = candidates.float().max(dim=-1)
#             traceback[t] = best_z.to(torch.int8)
        
#         # ===== 回溯 =====
#         current_states = alpha.argmax(dim=-1)
#         paths = torch.zeros(T, N, dtype=torch.int8, device=device)
#         batch_idx = torch.arange(N, device=device)
        
#         for t in range(T - 1, -1, -1):
#             best_edges = traceback[t, batch_idx, current_states]
#             paths[t] = best_edges
#             current_states = idx[current_states, best_edges.long()]

#         return paths.T.to(torch.long)