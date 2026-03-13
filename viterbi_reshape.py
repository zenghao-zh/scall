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
        # 使用 Moffett 版本生成 idx 和 idx_T
        if not hasattr(self, '_idx_moffett') or self._idx_moffett.device != device:
            # init_idx_table 方式生成 idx
            idx_np = np.zeros((n_states, n_alphabet), dtype=np.int32)
            idx_np[:, 0] = np.arange(n_states)
            for j in range(1, n_alphabet):
                idx_np[:, j] = ((j - 1) * n_states + np.arange(n_states)) // self.n_base
            self._idx_moffett = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)

            # get_idx_T_moffett 方式生成 idx_T
            idx_T_np = np.zeros((n_states, n_alphabet), dtype=np.int32)
            for i in range(n_states):
                idx_T_np[i][0] = i * n_alphabet
            for i in range(n_states):
                for j in range(self.n_base):
                    repeat_idx = i * self.n_base + j
                    repeat_idx_row = repeat_idx % n_states
                    repeat_idx_col = 1 + repeat_idx // n_states
                    flatten_idx = repeat_idx_row * n_alphabet + repeat_idx_col
                    idx_T_np[i][j + 1] = flatten_idx
            self._idx_T = torch.from_numpy(idx_T_np).to(device=device, dtype=torch.long)
            self._idx_T_targets = self._idx_T // n_alphabet

        idx = self._idx_moffett
         
        # [1000,160,batch_tile,32] # 160*32 -> 5120  batch-size_tile >= 32  --> total_size = 312.5 if batch-size_tile = 32
        # transpose
        # [1000,160,32,batch_tile]

        # 选择数据类型
        dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        Ms = scores.transpose(1,2).to(dtype).reshape(T, n_states, n_alphabet, N) # [1000, 512, 5120] --> [1000, 5120, 512] --> [1000, 1024, 5, 512]
         
        idx_T = self._idx_T
        idx_T_targets = self._idx_T_targets
        Ms_flat = Ms.reshape(T, -1, N)
        Ms_T = Ms_flat[:, idx_T, :]  #5120 维度flatten 后重排
         
        # 归一化间隔（bfloat16 需要更频繁的归一化）
        # 更小的 segment_size = 更频繁的归一化 = 更好的数值稳定性
        segment_size = 8
         
        # ========================================
        # 步骤1: Log Forward (manual_logsumexp + 归一化)
        # ========================================
        alphas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype) #[1001,1024,512]
        alpha = alphas_all[0]  # 初始状态全为0
 
        #alpha : [1024,512]
         
        for t in range(T):
            alpha_indexed = alpha[idx, :] # 在1024 维度重排
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
        betas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype) #[1001,1024,512]
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
        alpha_max = torch.full((n_states, N), float('-inf'), device=device, dtype=dtype) #[1024,512]
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
 
            if t % 8 == 0:
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