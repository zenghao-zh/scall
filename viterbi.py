import torch 
import numpy as np

# self.n_base = 4
# self.state_len = 5
# self.alphabet = ['N', 'A', 'C', 'G', 'T']
# scores.shape: [seqlen, batch_size, num_features] -> torch.Size([1000, 256, 5120])

class Decoder:
    def __init__(self, data_path='viterbi_inputs.pth', device = 'cuda:0'):
        data = torch.load(data_path)

        # 取出所有预计算值
        self.scores = data['scores'].to(device)
        self.idx = data['idx'].to(device, dtype=torch.long)
        self._idx_T = data['idx_T'].to(device)
        self._idx_T_targets = data['idx_T_targets'].to(device)
        self.n_base = data['n_base']
        self.state_len = data['state_len']
        self.alphabet = data['alphabet']
        self.true_paths = data['path'].to(device)

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
            
            idx_T = self._idx_T.to(device=device, dtype=torch.long)
            idx_T_targets = self._idx_T_targets.to(device=device, dtype=torch.long)
            
            # ===== 后向遍历 =====
            Ms_flat = Ms.reshape(T, N, -1)  # (T, N, C*NZ)
            Ms_T = Ms_flat[:, :, idx_T]  # 转置后的 Ms: (T, N, C, NZ)，Ms_T[t, n, s, j] 是从状态 s 出发的第 j 条边的分数
            
            beta_next = torch.zeros(N, n_states, device=device, dtype=dtype)
            betas_all = torch.zeros(T + 1, N, n_states, device=device, dtype=dtype)
            
            for t in range(T - 1, -1, -1):
                # candidates[n, s, j] = Ms_T[t, n, s, j] + beta[t+1, n, target(s,j)]
                # 即：从状态 s 发出第 j 条边的分数 + 到达目标状态后的剩余分数
                candidates = Ms_T[t] + beta_next[:, idx_T_targets]  # (N, C, NZ)
                
                # 精确 logsumexp
                beta_next = torch.logsumexp(candidates, dim=-1)
                betas_all[t] = beta_next
                
            # ===== 前向遍历 + 引导 =====
            # guided_Ms[t, n, c, z] = Ms[t, n, c, z] + beta[t+1, n, c]
            # 边分数 + 到达状态 c 后的最优剩余分数
            guided_Ms = Ms + betas_all[1:, :, :, None]
            
            alpha = torch.zeros(N, n_states, device=device, dtype=dtype)
            traceback = torch.zeros(T, N, n_states, dtype=torch.int8, device=device)
            
            for t in range(T):
                candidates = alpha[:, idx] + guided_Ms[t]
                alpha, best_z = candidates.max(dim=-1)
                traceback[t] = best_z.to(torch.int8)
            
            # ===== 回溯 =====
            current_states = alpha.argmax(dim=-1)
            paths = torch.zeros(T, N, dtype=torch.int8, device=device)
            batch_idx = torch.arange(N, device=device)
            
            for t in range(T - 1, -1, -1):
                best_edges = traceback[t, batch_idx, current_states]
                paths[t] = best_edges
                current_states = idx[current_states, best_edges.long()]
            
            return paths.T.to(torch.long)

if __name__ == '__main__':
    decoder = Decoder(data_path='viterbi_inputs.pth', device='cuda:0')
    paths = decoder.viterbi_fused_guided_fast(decoder.scores)

    if (paths == decoder.true_paths).all():
        print('test viterbi_fused_guided_fast passed')
    else:
        print('test viterbi_fused_guided_fast failed')
