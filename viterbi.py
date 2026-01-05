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

    def viterbi_fused_guided_fast(self, scores):
            """
            快速版带引导的 Viterbi（优化实现）
            
            使用转置索引正确计算后向分数：
            - idx[c, z] 表示到达状态 c 的第 z 条入边的来源状态
            - 后向计算需要从"出边"视角，因此需要对 idx 和 Ms 做转置
            """
            # scores = scores.to(torch.bfloat16)
            T, N, _ = scores.shape
            n_states = self.n_base ** self.state_len
            n_alphabet = len(self.alphabet)
            
            Ms = scores.reshape(T, N, n_states, n_alphabet)
            device = scores.device
            dtype = scores.dtype
            idx = self.idx.to(device=device, dtype=torch.long)
            
            idx_T = self._idx_T.to(device=device)
            idx_T_targets = self._idx_T_targets.to(device=device)
            
            # ===== 后向遍历 =====
            Ms_flat = Ms.reshape(T, N, -1)  # (T, N, C*NZ)
            # Ms_T = Ms_flat[:, :, idx_T]  # 转置后的 Ms: (T, N, C, NZ)，Ms_T[t, n, s, j] 是从状态 s 出发的第 j 条边的分数
            
            # 使用 mask 矩阵替代索引操作
            # 创建置换矩阵: mask[idx_T[c,z], c*NZ+z] = 1
            if not hasattr(self, '_idx_T_mask') or self._idx_T_mask.device != device:
                mask = torch.zeros(n_states * n_alphabet, n_states * n_alphabet, device=device, dtype=dtype)
                idx_T_flat = idx_T.flatten()  # (C*NZ,)
                src_indices = torch.arange(n_states * n_alphabet, device=device)
                mask[idx_T_flat, src_indices] = 1.0
                self._idx_T_mask = mask
            
            mask = self._idx_T_mask
            Ms_T = torch.matmul(Ms_flat, mask).reshape(T, N, n_states, n_alphabet)  # 转置后的 Ms: (T, N, C, NZ)，Ms_T[t, n, s, j] 是从状态 s 出发的第 j 条边的分数
            if self._onnx_exported:
                model = SimpleMatmul()
                model.weight.data = mask
                torch.onnx.export(
                    model,
                    (Ms_flat),
                    "huada_matmul_5120.onnx",
                    input_names=['input', 'weight'],
                    output_names=['output']
                )
                print(f"✓ Exported to matmul.onnx")
                # 保存输入输出测试数据
                # matmul_output = torch.matmul(Ms_flat, mask)
                # torch.save({
                #     'input': Ms_flat.cpu(),
                #     'weight': mask.cpu(),
                #     'output': matmul_output.cpu()
                # }, 'huada_matmul_5120.pth')
                # print(f"✓ Saved test data to matmul_test_data.pth")
                
                self._onnx_exported = False

                
            beta_next = torch.zeros(N, n_states, device=device, dtype=torch.float32)
            betas_all = torch.zeros(T + 1, N, n_states, device=device, dtype=torch.float32)
            
            for t in range(T - 1, -1, -1):
                # candidates[n, s, j] = Ms_T[t, n, s, j] + beta[t+1, n, target(s,j)]
                # 即：从状态 s 发出第 j 条边的分数 + 到达目标状态后的剩余分数
                # 使用 F.embedding 替代索引操作，避免 gather/index_select 的硬件兼容性问题
                # candidates = Ms_T[t] + beta_next[:, idx_T_targets]  # (N, C, NZ)
                beta_indexed = torch.nn.functional.embedding(idx_T_targets, beta_next.T).permute(2, 0, 1)
                candidates = Ms_T[t] + beta_indexed  # (N, C, NZ)
                
                # 精确 logsumexp
                beta_next = torch.logsumexp(candidates.float(), dim=-1)
                betas_all[t] = beta_next.to(dtype)
                
            # ===== 前向遍历 + 引导 =====
            # guided_Ms[t, n, c, z] = Ms[t, n, c, z] + beta[t+1, n, c]
            # 边分数 + 到达状态 c 后的最优剩余分数
            guided_Ms = Ms + betas_all[1:, :, :, None]
            
            alpha = torch.zeros(N, n_states, device=device, dtype=torch.float32)
            traceback = torch.zeros(T, N, n_states, dtype=torch.int8, device=device)
            
            for t in range(T):
                # 使用 F.embedding 替代 alpha[:, idx] 索引操作
                # alpha: (N, n_states), idx: (n_states, n_alphabet)
                # 输出: (N, n_states, n_alphabet)
                alpha_indexed = torch.nn.functional.embedding(idx, alpha.T).permute(2, 0, 1)
                candidates = alpha_indexed + guided_Ms[t]
                alpha, best_z = candidates.float().max(dim=-1)
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

    # 计算相似度（匹配率）
    similarity = (paths == decoder.true_paths).float().mean().item()
    print(f'Similarity: {similarity * 100:.2f}%')
    
    if similarity > 0.999:
        print(f'⚠️  test viterbi_fused_guided_fast passed ({similarity * 100:.2f}% match)')
    else:
        print(f'❌ test viterbi_fused_guided_fast failed ({similarity * 100:.2f}% match)')
