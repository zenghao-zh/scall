"""
Sample 139: detailed numerical comparison between GPU and SPU scores.
Shape: (960, 5120) → reshaped to (960, 1024, 5) for (T, n_states, n_alphabet).
"""

import os, sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_GPU = os.path.join(SCRIPT_DIR, "scores_gpu.pt")
FILE_SPU = os.path.join(SCRIPT_DIR, "spu_scores_512.pt")
SAMPLE = 139
ALPHABET = ['N(blank)', 'A', 'C', 'G', 'T']


def main():
    print("[Loading scores] ...")
    gpu_all = torch.load(FILE_GPU, map_location="cpu")
    spu_all = torch.load(FILE_SPU, map_location="cpu")

    # Extract sample 139: (T, C)
    gpu = gpu_all[:, SAMPLE, :].float()   # (960, 5120)
    spu = spu_all[:, SAMPLE, :].float()   # (960, 5120)
    T, C = gpu.shape
    n_states, n_alphabet = 1024, 5

    diff = gpu - spu
    abs_diff = diff.abs()

    # =========================================================
    # 1. 整体数值对比
    # =========================================================
    print(f"\n{'='*75}")
    print(f"Sample {SAMPLE} — 整体数值对比  (T={T}, C={C})")
    print(f"{'='*75}")
    print(f"  {'':25s} {'GPU(bf16)':>14s} {'SPU(fp16)':>14s}")
    print(f"  {'-'*53}")
    print(f"  {'min':25s} {gpu.min().item():>14.6f} {spu.min().item():>14.6f}")
    print(f"  {'max':25s} {gpu.max().item():>14.6f} {spu.max().item():>14.6f}")
    print(f"  {'mean':25s} {gpu.mean().item():>14.6f} {spu.mean().item():>14.6f}")
    print(f"  {'std':25s} {gpu.std().item():>14.6f} {spu.std().item():>14.6f}")

    print(f"\n  {'':25s} {'Value':>14s}")
    print(f"  {'-'*39}")
    print(f"  {'MAE':25s} {abs_diff.mean().item():>14.6e}")
    print(f"  {'Max AE':25s} {abs_diff.max().item():>14.6e}")
    print(f"  {'RMSE':25s} {diff.pow(2).mean().sqrt().item():>14.6e}")
    print(f"  {'Median AE':25s} {abs_diff.median().item():>14.6e}")

    cos = torch.nn.functional.cosine_similarity(gpu.reshape(1, -1), spu.reshape(1, -1)).item()
    print(f"  {'Cosine Similarity':25s} {cos:>14.10f}")

    total = abs_diff.numel()
    print(f"\n  --- 绝对误差分布 ---")
    for tol in [0, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        cnt = (abs_diff <= tol).sum().item()
        print(f"  |err| <= {tol:<6g}: {cnt:>9d}/{total} ({cnt/total*100:6.2f}%)")

    # =========================================================
    # 2. 按 timestep 统计
    # =========================================================
    per_t_mae = abs_diff.mean(dim=1)       # (960,)
    per_t_max = abs_diff.max(dim=1).values  # (960,)

    print(f"\n{'='*75}")
    print(f"按 timestep (T={T}) 统计")
    print(f"{'='*75}")
    print(f"  timestep MAE: min={per_t_mae.min().item():.6e}  max={per_t_mae.max().item():.6e}  mean={per_t_mae.mean().item():.6e}")
    print(f"  timestep Max: min={per_t_max.min().item():.6e}  max={per_t_max.max().item():.6e}  mean={per_t_max.mean().item():.6e}")

    # 按 MAE 分段统计
    bins = [0, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, float('inf')]
    print(f"\n  MAE 分段:")
    for i in range(len(bins)-1):
        mask = (per_t_mae >= bins[i]) & (per_t_mae < bins[i+1])
        cnt = mask.sum().item()
        hi = f"{bins[i+1]}" if bins[i+1] != float('inf') else "∞"
        print(f"    [{bins[i]:.2f}, {hi}): {cnt:>4d} timesteps ({cnt/T*100:.1f}%)")

    # Top 20 worst timesteps
    sorted_t = per_t_mae.argsort(descending=True)
    print(f"\n  Top 20 worst timesteps:")
    print(f"  {'t':>6s}  {'MAE':>12s}  {'MaxAE':>12s}  {'>0.5 cnt':>10s}  {'>1.0 cnt':>10s}  {'>2.0 cnt':>10s}")
    print(f"  {'-'*68}")
    for rank in range(20):
        t = sorted_t[rank].item()
        row_diff = abs_diff[t]
        print(f"  {t:>6d}  {per_t_mae[t].item():>12.6e}  {per_t_max[t].item():>12.6e}  "
              f"{(row_diff > 0.5).sum().item():>10d}  "
              f"{(row_diff > 1.0).sum().item():>10d}  "
              f"{(row_diff > 2.0).sum().item():>10d}")

    # =========================================================
    # 3. reshape 成 (T, 1024, 5) 按 state 和 alphabet 看
    # =========================================================
    gpu_r = gpu.reshape(T, n_states, n_alphabet)
    spu_r = spu.reshape(T, n_states, n_alphabet)
    diff_r = (gpu_r - spu_r).abs()

    print(f"\n{'='*75}")
    print(f"按 alphabet 通道 (N/A/C/G/T) 统计")
    print(f"{'='*75}")
    print(f"  {'Channel':>12s}  {'MAE':>12s}  {'MaxAE':>12s}  {'RMSE':>12s}  {'>1.0 cnt':>10s}")
    print(f"  {'-'*62}")
    for ch in range(n_alphabet):
        ch_diff = diff_r[:, :, ch]
        ch_mae = ch_diff.mean().item()
        ch_max = ch_diff.max().item()
        ch_rmse = (gpu_r[:,:,ch] - spu_r[:,:,ch]).pow(2).mean().sqrt().item()
        ch_gt1 = (ch_diff > 1.0).sum().item()
        print(f"  {ALPHABET[ch]:>12s}  {ch_mae:>12.6e}  {ch_max:>12.6e}  {ch_rmse:>12.6e}  {ch_gt1:>10d}")

    # =========================================================
    # 4. 按 state 统计，找 top 差异 state
    # =========================================================
    per_state_mae = diff_r.mean(dim=(0, 2))  # (1024,)
    sorted_states = per_state_mae.argsort(descending=True)

    print(f"\n{'='*75}")
    print(f"按 state (1024 个) 统计 — Top 20 差异最大")
    print(f"{'='*75}")
    print(f"  {'Rank':>6s}  {'State':>6s}  {'MAE':>12s}  {'MaxAE':>12s}")
    print(f"  {'-'*42}")
    for rank in range(20):
        st = sorted_states[rank].item()
        st_diff = diff_r[:, st, :]
        print(f"  {rank+1:>6d}  {st:>6d}  {st_diff.mean().item():>12.6e}  {st_diff.max().item():>12.6e}")

    # =========================================================
    # 5. 找到绝对误差 > 2 的所有位置，详细列出
    # =========================================================
    large_mask = diff_r > 2.0
    large_count = large_mask.sum().item()
    print(f"\n{'='*75}")
    print(f"绝对误差 > 2.0 的位置 (共 {large_count} 个)")
    print(f"{'='*75}")
    if large_count > 0 and large_count <= 200:
        positions = large_mask.nonzero(as_tuple=False)  # (K, 3): t, state, ch
        print(f"  {'t':>6s}  {'state':>6s}  {'ch':>10s}  {'GPU':>10s}  {'SPU':>10s}  {'diff':>10s}")
        print(f"  {'-'*58}")
        for k in range(positions.shape[0]):
            t, st, ch = positions[k].tolist()
            g = gpu_r[t, st, ch].item()
            s = spu_r[t, st, ch].item()
            print(f"  {t:>6d}  {st:>6d}  {ALPHABET[ch]:>10s}  {g:>10.4f}  {s:>10.4f}  {g-s:>+10.4f}")
    elif large_count > 200:
        print(f"  太多了({large_count})，只打印前 50 个:")
        positions = large_mask.nonzero(as_tuple=False)
        print(f"  {'t':>6s}  {'state':>6s}  {'ch':>10s}  {'GPU':>10s}  {'SPU':>10s}  {'diff':>10s}")
        print(f"  {'-'*58}")
        for k in range(50):
            t, st, ch = positions[k].tolist()
            g = gpu_r[t, st, ch].item()
            s = spu_r[t, st, ch].item()
            print(f"  {t:>6d}  {st:>6d}  {ALPHABET[ch]:>10s}  {g:>10.4f}  {s:>10.4f}  {g-s:>+10.4f}")
        # 统计这些大误差的分布
        ts = positions[:, 0].numpy()
        print(f"\n  大误差 timestep 分布:")
        t_unique, t_counts = np.unique(ts, return_counts=True)
        sorted_tc = np.argsort(-t_counts)
        for i in range(min(20, len(t_unique))):
            idx = sorted_tc[i]
            print(f"    t={t_unique[idx]:<5d}: {t_counts[idx]:>5d} 个大误差")

    # =========================================================
    # 6. 每 100 个 timestep 的平均 MAE 趋势
    # =========================================================
    print(f"\n{'='*75}")
    print(f"按 timestep 段 (每 100 步) 的 MAE 趋势")
    print(f"{'='*75}")
    print(f"  {'Range':>12s}  {'MAE':>12s}  {'MaxAE':>12s}  {'>0.5 cnt':>10s}  {'>2.0 cnt':>10s}")
    print(f"  {'-'*62}")
    for start in range(0, T, 100):
        end = min(start + 100, T)
        seg = abs_diff[start:end]
        seg_mae = seg.mean().item()
        seg_max = seg.max().item()
        seg_05 = (seg > 0.5).sum().item()
        seg_20 = (seg > 2.0).sum().item()
        print(f"  {f't[{start}:{end}]':>12s}  {seg_mae:>12.6e}  {seg_max:>12.6e}  {seg_05:>10d}  {seg_20:>10d}")

    # =========================================================
    # 7. GPU 和 SPU 的整体值分布 (直方图)
    # =========================================================
    gpu_r = gpu.reshape(T, n_states, n_alphabet)
    spu_r = spu.reshape(T, n_states, n_alphabet)

    print(f"\n{'='*75}")
    print(f"GPU 和 SPU 整体值分布 (所有通道)")
    print(f"{'='*75}")
    edges = [-6, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5,
             0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6]
    gpu_flat = gpu.reshape(-1)
    spu_flat = spu.reshape(-1)
    total_all = gpu_flat.numel()
    print(f"  {'Bin':>16s}  {'GPU count':>12s} {'GPU%':>8s}  {'SPU count':>12s} {'SPU%':>8s}  {'diff%':>8s}")
    print(f"  {'-'*72}")
    for i in range(len(edges)-1):
        g_cnt = ((gpu_flat >= edges[i]) & (gpu_flat < edges[i+1])).sum().item()
        s_cnt = ((spu_flat >= edges[i]) & (spu_flat < edges[i+1])).sum().item()
        g_pct = g_cnt / total_all * 100
        s_pct = s_cnt / total_all * 100
        bar_g = '█' * int(g_pct / 2)
        bar_s = '▒' * int(s_pct / 2)
        print(f"  [{edges[i]:>5.1f},{edges[i+1]:>5.1f})"
              f"  {g_cnt:>12d} {g_pct:>7.2f}%  {s_cnt:>12d} {s_pct:>7.2f}%  {s_pct-g_pct:>+7.2f}%")

    # 分通道打印: blank vs 非 blank
    print(f"\n{'='*75}")
    print(f"Blank 通道 (ch=0) 分布")
    print(f"{'='*75}")
    g_blank = gpu_r[:, :, 0].reshape(-1)
    s_blank = spu_r[:, :, 0].reshape(-1)
    g_unique = torch.unique(g_blank)
    s_unique = torch.unique(s_blank)
    print(f"  GPU blank unique values ({len(g_unique)}): {g_unique.tolist()[:10]}")
    print(f"  SPU blank unique values ({len(s_unique)}): {s_unique.tolist()[:10]}")

    print(f"\n{'='*75}")
    print(f"非 Blank 通道 (A/C/G/T) 值分布")
    print(f"{'='*75}")
    g_nb = gpu_r[:, :, 1:].reshape(-1)
    s_nb = spu_r[:, :, 1:].reshape(-1)
    total_nb = g_nb.numel()
    print(f"  {'Bin':>16s}  {'GPU count':>12s} {'GPU%':>8s}  {'SPU count':>12s} {'SPU%':>8s}  {'diff%':>8s}")
    print(f"  {'-'*72}")
    for i in range(len(edges)-1):
        g_cnt = ((g_nb >= edges[i]) & (g_nb < edges[i+1])).sum().item()
        s_cnt = ((s_nb >= edges[i]) & (s_nb < edges[i+1])).sum().item()
        g_pct = g_cnt / total_nb * 100
        s_pct = s_cnt / total_nb * 100
        print(f"  [{edges[i]:>5.1f},{edges[i+1]:>5.1f})"
              f"  {g_cnt:>12d} {g_pct:>7.2f}%  {s_cnt:>12d} {s_pct:>7.2f}%  {s_pct-g_pct:>+7.2f}%")

    # 分 A/C/G/T 各通道分布
    for ch in range(1, n_alphabet):
        print(f"\n{'='*75}")
        print(f"通道 {ALPHABET[ch]} 值分布")
        print(f"{'='*75}")
        g_ch = gpu_r[:, :, ch].reshape(-1)
        s_ch = spu_r[:, :, ch].reshape(-1)
        total_ch = g_ch.numel()
        print(f"  GPU: min={g_ch.min().item():.4f}  max={g_ch.max().item():.4f}  mean={g_ch.mean().item():.4f}  std={g_ch.std().item():.4f}")
        print(f"  SPU: min={s_ch.min().item():.4f}  max={s_ch.max().item():.4f}  mean={s_ch.mean().item():.4f}  std={s_ch.std().item():.4f}")
        print(f"  {'Bin':>16s}  {'GPU count':>12s} {'GPU%':>8s}  {'SPU count':>12s} {'SPU%':>8s}  {'diff%':>8s}")
        print(f"  {'-'*72}")
        for i in range(len(edges)-1):
            g_cnt = ((g_ch >= edges[i]) & (g_ch < edges[i+1])).sum().item()
            s_cnt = ((s_ch >= edges[i]) & (s_ch < edges[i+1])).sum().item()
            g_pct = g_cnt / total_ch * 100
            s_pct = s_cnt / total_ch * 100
            print(f"  [{edges[i]:>5.1f},{edges[i+1]:>5.1f})"
                  f"  {g_cnt:>12d} {g_pct:>7.2f}%  {s_cnt:>12d} {s_pct:>7.2f}%  {s_pct-g_pct:>+7.2f}%")

    # GPU 和 SPU 的 percentile 对比
    print(f"\n{'='*75}")
    print(f"非 Blank 通道分位数对比")
    print(f"{'='*75}")
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    print(f"  {'Percentile':>12s}  {'GPU':>12s}  {'SPU':>12s}  {'diff':>10s}")
    print(f"  {'-'*50}")
    for p in percentiles:
        if p == 0:
            g_val = g_nb.min().item()
            s_val = s_nb.min().item()
        elif p == 100:
            g_val = g_nb.max().item()
            s_val = s_nb.max().item()
        else:
            g_val = torch.quantile(g_nb, p/100.0).item()
            s_val = torch.quantile(s_nb, p/100.0).item()
        print(f"  {p:>11d}%  {g_val:>12.4f}  {s_val:>12.4f}  {s_val-g_val:>+10.4f}")

    print(f"\n{'='*75}")
    print("Done.")


if __name__ == "__main__":
    main()
