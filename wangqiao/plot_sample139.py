"""
Sample 139: 画图对比 GPU vs SPU scores 的分布和差异。
"""

import os, sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_GPU = os.path.join(SCRIPT_DIR, "scores_gpu.pt")
FILE_SPU = os.path.join(SCRIPT_DIR, "spu_scores_512.pt")
SAMPLE = 314
ALPHABET = ['N(blank)', 'A', 'C', 'G', 'T']
COLORS = {'GPU': '#2196F3', 'SPU': '#FF5722'}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})


def main():
    print("[Loading scores] ...")
    gpu_all = torch.load(FILE_GPU, map_location="cpu")
    spu_all = torch.load(FILE_SPU, map_location="cpu")

    gpu = gpu_all[:, SAMPLE, :].to(torch.bfloat16).float()  # (960, 5120)
    spu = spu_all[:, SAMPLE, :].to(torch.bfloat16).float()  # SPU 也是 bf16
    T, C = gpu.shape
    n_states, n_alphabet = 1024, 5

    gpu_r = gpu.reshape(T, n_states, n_alphabet)
    spu_r = spu.reshape(T, n_states, n_alphabet)
    diff = gpu - spu
    abs_diff = diff.abs()

    # =========================================================
    # Figure 1: 整体分布对比 (6 子图)
    # =========================================================
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'Sample {SAMPLE} — GPU vs SPU Score Distribution', fontsize=16, fontweight='bold')
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    bins = np.linspace(-5.5, 5.5, 80)

    # --- 1a: 非 blank 整体分布 ---
    ax = fig.add_subplot(gs[0, 0])
    g_nb = gpu_r[:, :, 1:].reshape(-1).numpy()
    s_nb = spu_r[:, :, 1:].reshape(-1).numpy()
    ax.hist(g_nb, bins=bins, alpha=0.6, color=COLORS['GPU'], label='GPU (bf16)', density=True)
    ax.hist(s_nb, bins=bins, alpha=0.6, color=COLORS['SPU'], label='SPU (bf16)', density=True)
    ax.set_title('Non-blank channels (A/C/G/T) distribution')
    ax.set_xlabel('Score value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, label='blank=2.0')

    # --- 1b: 分通道分布 (叠加) ---
    ax = fig.add_subplot(gs[0, 1])
    ch_colors = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63']
    for ch_idx, ch in enumerate(range(1, n_alphabet)):
        g_ch = gpu_r[:, :, ch].reshape(-1).numpy()
        s_ch = spu_r[:, :, ch].reshape(-1).numpy()
        ax.hist(g_ch, bins=bins, alpha=0.3, color=ch_colors[ch_idx],
                density=True, histtype='step', linewidth=2,
                label=f'{ALPHABET[ch]} GPU')
        ax.hist(s_ch, bins=bins, alpha=0.3, color=ch_colors[ch_idx],
                density=True, histtype='step', linewidth=2, linestyle='--',
                label=f'{ALPHABET[ch]} SPU')
    ax.set_title('Per-channel distribution (solid=GPU, dashed=SPU)')
    ax.set_xlabel('Score value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8, ncol=2)

    # --- 1c: GPU vs SPU 散点图 (采样) ---
    ax = fig.add_subplot(gs[1, 0])
    n_sample = min(200000, len(g_nb))
    idx = np.random.choice(len(g_nb), n_sample, replace=False)
    ax.scatter(g_nb[idx], s_nb[idx], alpha=0.02, s=1, c='steelblue')
    lims = [-5.5, 5.5]
    ax.plot(lims, lims, 'r-', linewidth=1, label='y=x')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('GPU vs SPU score scatter (non-blank, sampled)')
    ax.set_xlabel('GPU score')
    ax.set_ylabel('SPU score')
    ax.legend()
    ax.set_aspect('equal')

    # --- 1d: 误差分布 ---
    ax = fig.add_subplot(gs[1, 1])
    err = (gpu_r[:, :, 1:] - spu_r[:, :, 1:]).reshape(-1).numpy()
    ax.hist(err, bins=np.linspace(-6, 6, 100), alpha=0.7, color='#9C27B0', density=True)
    ax.set_title('Error distribution (GPU - SPU, non-blank)')
    ax.set_xlabel('GPU - SPU')
    ax.set_ylabel('Density')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    mean_err = np.mean(err)
    ax.axvline(x=mean_err, color='orange', linestyle='--', alpha=0.7)
    ax.text(mean_err + 0.1, ax.get_ylim()[1] * 0.9, f'mean={mean_err:.3f}',
            color='orange', fontsize=10)

    # --- 1e: Per-timestep MAE ---
    ax = fig.add_subplot(gs[2, 0])
    per_t_mae = abs_diff.mean(dim=1).numpy()
    ax.plot(range(T), per_t_mae, color='#E91E63', linewidth=0.8)
    ax.fill_between(range(T), per_t_mae, alpha=0.3, color='#E91E63')
    ax.set_title('Per-timestep MAE')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('MAE')
    # Mark worst region
    worst_t = np.argmax(per_t_mae)
    ax.annotate(f't={worst_t}\nMAE={per_t_mae[worst_t]:.3f}',
                xy=(worst_t, per_t_mae[worst_t]),
                xytext=(worst_t - 150, per_t_mae[worst_t]),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=9, color='red')

    # --- 1f: 分位数 QQ 图 ---
    ax = fig.add_subplot(gs[2, 1])
    pcts = np.arange(0, 101, 1)
    g_quantiles = np.percentile(g_nb, pcts)
    s_quantiles = np.percentile(s_nb, pcts)
    ax.plot(g_quantiles, s_quantiles, 'o-', color='#009688', markersize=3)
    ax.plot([-5.5, 5.5], [-5.5, 5.5], 'r--', linewidth=1, label='y=x')
    ax.set_title('QQ plot (GPU quantiles vs SPU quantiles)')
    ax.set_xlabel('GPU quantile')
    ax.set_ylabel('SPU quantile')
    ax.legend()
    ax.set_aspect('equal')
    # 标注关键分位点
    for p in [50, 90, 95, 99]:
        idx_p = p
        ax.annotate(f'P{p}', xy=(g_quantiles[idx_p], s_quantiles[idx_p]),
                    fontsize=8, color='red',
                    xytext=(5, 5), textcoords='offset points')

    out1 = os.path.join(SCRIPT_DIR, "sample314_distribution.png")
    fig.savefig(out1, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out1}")
    plt.close(fig)

    # =========================================================
    # Figure 2: 2D heatmap — score 差异 (T × C_reshaped)
    # =========================================================
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f'Sample {SAMPLE} — Score Difference Heatmaps (GPU - SPU)', fontsize=16, fontweight='bold')

    for ch_idx, ch in enumerate(range(1, n_alphabet)):
        ax = axes[ch_idx // 2, ch_idx % 2]
        diff_ch = (gpu_r[:, :, ch] - spu_r[:, :, ch]).numpy()  # (960, 1024)
        # Downsample states for visibility
        im = ax.imshow(diff_ch.T, aspect='auto', cmap='RdBu_r',
                        vmin=-3, vmax=3, interpolation='nearest')
        ax.set_title(f'Channel {ALPHABET[ch]}  (GPU - SPU)')
        ax.set_xlabel('Timestep t')
        ax.set_ylabel('State')
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    plt.tight_layout()
    out2 = os.path.join(SCRIPT_DIR, "sample314_heatmap.png")
    fig.savefig(out2, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out2}")
    plt.close(fig)

    # =========================================================
    # Figure 3: 正值区放大 + CDF 对比
    # =========================================================
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Sample {SAMPLE} — Positive Score Region & CDF', fontsize=16, fontweight='bold')

    # 3a: 正值区域放大
    ax = axes[0]
    pos_bins = np.linspace(-1, 5, 60)
    ax.hist(g_nb[g_nb > -1], bins=pos_bins, alpha=0.6, color=COLORS['GPU'],
            label='GPU', density=True)
    ax.hist(s_nb[s_nb > -1], bins=pos_bins, alpha=0.6, color=COLORS['SPU'],
            label='SPU', density=True)
    ax.set_title('Positive region zoom (score > -1)')
    ax.set_xlabel('Score value')
    ax.set_ylabel('Density')
    ax.legend()
    ax.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5)
    ax.text(2.05, ax.get_ylim()[1] * 0.9, 'blank=2.0', fontsize=9, color='gray')

    # 3b: CDF 对比
    ax = axes[1]
    g_sorted = np.sort(g_nb)
    s_sorted = np.sort(s_nb)
    n = len(g_sorted)
    cdf_y = np.arange(1, n + 1) / n
    # Subsample for plotting speed
    step = max(1, n // 5000)
    ax.plot(g_sorted[::step], cdf_y[::step], color=COLORS['GPU'], linewidth=1.5, label='GPU')
    ax.plot(s_sorted[::step], cdf_y[::step], color=COLORS['SPU'], linewidth=1.5, label='SPU')
    ax.set_title('CDF of non-blank scores')
    ax.set_xlabel('Score value')
    ax.set_ylabel('CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3c: CDF 差 (SPU CDF - GPU CDF at same score values)
    ax = axes[2]
    eval_points = np.linspace(-5, 5, 500)
    g_cdf = np.searchsorted(g_sorted, eval_points) / n
    s_cdf = np.searchsorted(s_sorted, eval_points) / n
    cdf_diff = s_cdf - g_cdf
    ax.plot(eval_points, cdf_diff * 100, color='#9C27B0', linewidth=1.5)
    ax.fill_between(eval_points, cdf_diff * 100, alpha=0.3, color='#9C27B0')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_title('CDF difference (SPU - GPU)')
    ax.set_xlabel('Score value')
    ax.set_ylabel('CDF diff (%)')
    ax.grid(True, alpha=0.3)
    # Annotate peak
    peak_idx = np.argmax(np.abs(cdf_diff))
    ax.annotate(f'peak={cdf_diff[peak_idx]*100:.2f}%\n@score={eval_points[peak_idx]:.2f}',
                xy=(eval_points[peak_idx], cdf_diff[peak_idx] * 100),
                xytext=(eval_points[peak_idx] + 1, cdf_diff[peak_idx] * 100 + 0.5),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')

    plt.tight_layout()
    out3 = os.path.join(SCRIPT_DIR, "sample314_cdf.png")
    fig.savefig(out3, dpi=150, bbox_inches='tight')
    print(f"[Saved] {out3}")
    plt.close(fig)

    print("\nDone. Generated 3 figures:")
    print(f"  1. {out1}")
    print(f"  2. {out2}")
    print(f"  3. {out3}")


if __name__ == "__main__":
    main()
