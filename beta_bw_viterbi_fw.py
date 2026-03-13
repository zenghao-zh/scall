import torch
import numpy as np


def manual_logsumexp(x, dim=-1, keepdim=False):
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_shifted = x - x_max
    result = x_max + torch.log2(torch.sum(torch.exp(x_shifted), dim=dim, keepdim=True))
    if not keepdim:
        result = result.squeeze(dim)
    return result


def torch_beta_backward_reference(ms_score, n_base=4, state_len=5, segment_size=8, use_bfloat16=True):
    """
    只计算 backward pass 的 betas_all。

    Args:
        ms_score: CRF encoder 的输出, shape (T=960, N=512, C=5120), C = n_states * n_alphabet
        n_base: 碱基数量，默认 4
        state_len: 状态长度，默认 5
        segment_size: 归一化间隔，默认 8
        use_bfloat16: 是否使用 bfloat16

    Returns:
        betas_all: shape (961, 1024, 512)
    """
    T, N, C = ms_score.shape                # T=960, N=512, C=5120
    n_states = n_base ** state_len          # 4^5 = 1024
    n_alphabet = n_base + 1                 # 4+1 = 5
    device = ms_score.device
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    # ========================================
    # 计算 idx_T 和 idx_T_targets（backward 索引表）
    # ========================================
    idx_T = np.zeros((n_states, n_alphabet), dtype=np.int32)       # (1024, 5)
    for i in range(n_states):
        idx_T[i][0] = i * n_alphabet
    for i in range(n_states):
        for j in range(n_base):
            repeat_idx = i * n_base + j
            repeat_idx_row = repeat_idx % n_states
            repeat_idx_col = 1 + repeat_idx // n_states
            flatten_idx = repeat_idx_row * n_alphabet + repeat_idx_col
            idx_T[i][j + 1] = flatten_idx
    idx_T = torch.from_numpy(idx_T).to(device=device, dtype=torch.long)  # (1024, 5)
    idx_T_targets = idx_T // n_alphabet                                   # (1024, 5)

    # ========================================
    # 将 CRF encoder 输出变换为 Ms_T
    # ========================================
    # ms_score:  (960, 512, 5120)
    # transpose: (960, 5120, 512)
    # reshape:   (960, 1024, 5, 512)
    Ms = ms_score.transpose(1, 2).to(dtype).reshape(T, n_states, n_alphabet, N)  # (960, 1024, 5, 512)
    Ms_flat = Ms.reshape(T, -1, N)          # (960, 5120, 512)
    Ms_T = Ms_flat[:, idx_T, :]             # (960, 1024, 5, 512)  按 idx_T 重排

    # ========================================
    # Backward pass: 计算 betas_all
    # ========================================
    betas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype)  # (961, 1024, 512)
    beta = betas_all[T]                     # (1024, 512)  终止状态全为 0

    for t in range(T - 1, -1, -1):          # t = 959, 958, ..., 0
        beta_indexed = beta[idx_T_targets, :]   # (1024, 5, 512)  按目标状态索引 beta
        candidates = Ms_T[t] + beta_indexed     # (1024, 5, 512)  转移分数 + beta
        beta = manual_logsumexp(candidates, dim=1)  # (1024, 512)  在 alphabet 维度求 logsumexp

        if t % segment_size == 0:
            beta_min = beta.min(dim=0, keepdim=True)[0]  # (1, 512)
            beta = beta - beta_min                        # (1024, 512)

        betas_all[t] = beta                 # (1024, 512) 写入 betas_all[t]

    return betas_all                        # (961, 1024, 512)

def torch_viterbi_forward_reference(alphas_all, betas_all, ms_score,
                                    n_base=4, state_len=5, segment_size=8, use_bfloat16=True):
    """
    Posterior-guided Viterbi forward + traceback。

    Args:
        alphas_all: forward 结果, shape (961, 1024, 512)
        betas_all:  backward 结果, shape (961, 1024, 512)
        ms_score:   CRF encoder 输出, shape (960, 512, 5120)

    Returns:
        paths: 解码路径, shape (512, 960)
    """
    T, N, C = ms_score.shape                # T=960, N=512, C=5120
    n_states = n_base ** state_len          # 4^5 = 1024
    n_alphabet = n_base + 1                 # 4+1 = 5
    device = ms_score.device
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    # ========================================
    # 计算 idx（forward 索引表）
    # ========================================
    idx = np.zeros((n_states, n_alphabet), dtype=np.int32)         # (1024, 5)
    idx[:, 0] = np.arange(n_states)                                # blank: 自身 -> 自身
    for j in range(1, n_alphabet):                                 # j = 1,2,3,4
        idx[:, j] = ((j - 1) * n_states + np.arange(n_states)) // n_base
    idx = torch.from_numpy(idx).to(device=device, dtype=torch.long)  # (1024, 5)

    # ========================================
    # 将 CRF encoder 输出变换为 Ms
    # ========================================
    # ms_score:  (960, 512, 5120)
    # transpose: (960, 5120, 512)
    # reshape:   (960, 1024, 5, 512)
    Ms = ms_score.transpose(1, 2).to(dtype).reshape(T, n_states, n_alphabet, N)  # (960, 1024, 5, 512)

    # ========================================
    # Posterior-guided Viterbi forward
    # ========================================
    alpha_max = torch.full((n_states, N), float('-inf'), device=device, dtype=dtype)  # (1024, 512)
    alpha_max[0, :] = 0.0
    traceback = torch.zeros(T, n_states, N, dtype=torch.int8, device=device)  # (960, 1024, 512)

    for t in range(T):                          # t = 0, 1, ..., 959
        # guided_scores = alpha[src] + Ms + beta[dst]
        alpha_indexed_src = alphas_all[t][idx, :]           # (1024, 5, 512)  按 idx 索引 alpha
        beta_indexed_dst = betas_all[t + 1][:, None, :]    # (1024, 1, 512)  广播到 alphabet 维
        guided_scores_t = alpha_indexed_src + Ms[t] + beta_indexed_dst  # (1024, 5, 512)

        # Viterbi: alpha_max[dst] = max over alphabet(alpha_max[src] + guided_scores)
        alpha_indexed_max = alpha_max[idx, :]               # (1024, 5, 512)  按 idx 索引 alpha_max
        candidates = alpha_indexed_max + guided_scores_t    # (1024, 5, 512)
        alpha_max, best_z = candidates.max(dim=1)           # alpha_max: (1024, 512), best_z: (1024, 512)
        traceback[t] = best_z.to(torch.int8)                # (1024, 512) 写入 traceback

        if t % segment_size == 0:
            alpha_max = alpha_max - alpha_max.max(dim=0, keepdim=True)[0]  # (1024, 512) 归一化

    return alpha_max, traceback