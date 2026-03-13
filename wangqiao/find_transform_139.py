"""
Sample 139: 分析 GPU vs SPU 的数值关系，尝试多种变换提升相似度，
并用 Viterbi 解码验证精度是否提升。
"""

import os, sys, time, random
import torch
import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

import toml
from viterbi_0224 import CTC_CRF, decode_ref, accuracy, TrainingDataSet3

FILE_GPU = os.path.join(SCRIPT_DIR, "scores_gpu.pt")
FILE_SPU = os.path.join(SCRIPT_DIR, "spu_scores_512.pt")
CONFIG_PATH = "/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/config.toml"
DATA_DIR = "/workspace/huada/moffett_data/250F600274011_train_data/val"
DEVICE = "cuda:0"
SAMPLE = 139
ALPHABET = ['N(blank)', 'A', 'C', 'G', 'T']


def decode_single(seqdist, scores_single, device):
    """Viterbi decode on (T, 1, C)."""
    scores = scores_single.to(device)
    with torch.no_grad():
        paths = seqdist.viterbi_guided_bidirectional_reshape(
            scores.to(torch.bfloat16), use_bfloat16=True
        )
    return seqdist.path_to_str(paths[0].cpu().numpy())


def eval_transform(name, spu_transformed, gpu, seqdist, ref, device):
    """Evaluate a transformation: compute MAE, cosine sim, then decode & accuracy."""
    diff = (gpu.float() - spu_transformed.float()).abs()
    mae = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        gpu.float().reshape(1, -1), spu_transformed.float().reshape(1, -1)
    ).item()
    max_ae = diff.max().item()

    # Decode
    T, C = spu_transformed.shape
    s = spu_transformed.unsqueeze(1)  # (T, 1, C)
    seq = decode_single(seqdist, s, device)
    acc = accuracy(ref, seq, min_coverage=0.95) if len(seq) else 0.0

    return {"name": name, "mae": mae, "cos": cos, "max_ae": max_ae,
            "acc": acc, "seq_len": len(seq)}


def main():
    torch.manual_seed(25)
    torch.cuda.manual_seed_all(25)
    np.random.seed(25)
    random.seed(25)

    config = toml.load(CONFIG_PATH)
    seqdist = CTC_CRF(state_len=config["global_norm"]["state_len"],
                       alphabet=config["labels"]["labels"])

    # Load target
    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(DATA_DIR, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    for data, target, *_ in loader:
        targets = list(torch.unbind(target, 0))
        break
    ref = decode_ref(targets[SAMPLE], config["labels"]["labels"])

    # Load scores
    print("[Loading scores] ...")
    gpu_all = torch.load(FILE_GPU, map_location="cpu")
    spu_all = torch.load(FILE_SPU, map_location="cpu")

    gpu = gpu_all[:, SAMPLE, :].float()  # (960, 5120)
    spu = spu_all[:, SAMPLE, :].float()  # (960, 5120)
    T, C = gpu.shape
    n_states, n_alphabet = 1024, 5

    # =========================================================
    # 0. Baseline: GPU accuracy & original SPU accuracy
    # =========================================================
    print(f"\n[Reference] length={len(ref)}")
    gpu_seq = decode_single(seqdist, gpu.unsqueeze(1), DEVICE)
    gpu_acc = accuracy(ref, gpu_seq, min_coverage=0.95) if len(gpu_seq) else 0.0
    print(f"[GPU baseline] acc={gpu_acc:.4f}%, seq_len={len(gpu_seq)}")

    # =========================================================
    # 1. 分析 GPU vs SPU 数值关系
    # =========================================================
    gpu_r = gpu.reshape(T, n_states, n_alphabet)
    spu_r = spu.reshape(T, n_states, n_alphabet)

    # 分通道分析: blank (ch=0) 和非 blank (ch=1~4)
    print(f"\n{'='*75}")
    print("数值关系分析 (非 blank 通道)")
    print(f"{'='*75}")

    gpu_nonblank = gpu_r[:, :, 1:].reshape(-1).numpy()
    spu_nonblank = spu_r[:, :, 1:].reshape(-1).numpy()

    # 线性回归: GPU = a * SPU + b
    slope, intercept, r_value, _, _ = sp_stats.linregress(spu_nonblank, gpu_nonblank)
    print(f"  线性回归: GPU = {slope:.6f} * SPU + ({intercept:.6f})")
    print(f"  R² = {r_value**2:.6f}")

    # 分区间统计
    print(f"\n  分区间统计 (SPU value → GPU mean):")
    bins = [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
    for i in range(len(bins)-1):
        mask = (spu_nonblank >= bins[i]) & (spu_nonblank < bins[i+1])
        if mask.sum() > 0:
            g_in_bin = gpu_nonblank[mask]
            s_in_bin = spu_nonblank[mask]
            print(f"    SPU ∈ [{bins[i]:>3d}, {bins[i+1]:>3d}): "
                  f"count={mask.sum():>8d}  GPU_mean={g_in_bin.mean():>7.3f}  "
                  f"GPU_std={g_in_bin.std():>6.3f}  SPU_mean={s_in_bin.mean():>7.3f}")

    # 分通道线性回归
    print(f"\n  分通道线性回归:")
    ch_params = {}
    for ch in range(1, n_alphabet):
        g_ch = gpu_r[:, :, ch].reshape(-1).numpy()
        s_ch = spu_r[:, :, ch].reshape(-1).numpy()
        sl, ic, rv, _, _ = sp_stats.linregress(s_ch, g_ch)
        ch_params[ch] = (sl, ic)
        print(f"    {ALPHABET[ch]}: GPU = {sl:.6f} * SPU + ({ic:.6f}), R²={rv**2:.6f}")

    # =========================================================
    # 2. 尝试各种变换
    # =========================================================
    print(f"\n{'='*75}")
    print("尝试各种变换")
    print(f"{'='*75}")

    results = []

    # --- 0: Original SPU (baseline) ---
    res = eval_transform("0_original_SPU", spu, gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- 1: Clip SPU < threshold to -5 ---
    for thresh in [-4.0, -3.5, -3.0, -2.5, -2.0]:
        spu_clip = spu_r.clone()
        spu_clip[:, :, 1:] = torch.where(spu_clip[:, :, 1:] < thresh,
                                           torch.tensor(-5.0), spu_clip[:, :, 1:])
        res = eval_transform(f"1_clip_below_{thresh}_to_-5",
                             spu_clip.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- 2: 全局线性变换 (非 blank): y = slope * x + intercept ---
    spu_lr = spu_r.clone()
    spu_lr[:, :, 1:] = slope * spu_lr[:, :, 1:] + intercept
    res = eval_transform("2_global_linear_regression",
                         spu_lr.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- 3: 分通道线性变换 ---
    spu_ch_lr = spu_r.clone()
    for ch in range(1, n_alphabet):
        sl, ic = ch_params[ch]
        spu_ch_lr[:, :, ch] = sl * spu_ch_lr[:, :, ch] + ic
    res = eval_transform("3_per_channel_linear_regression",
                         spu_ch_lr.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- 4: Scale 非 blank 通道 (乘以缩放系数) ---
    for scale_f in [1.05, 1.1, 1.15, 1.2, 1.3]:
        spu_sc = spu_r.clone()
        spu_sc[:, :, 1:] = spu_sc[:, :, 1:] * scale_f
        res = eval_transform(f"4_scale_nonblank_x{scale_f}",
                             spu_sc.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- 5: Clip + Scale 组合 ---
    for thresh in [-3.0, -2.5]:
        for scale_f in [1.1, 1.2]:
            spu_cs = spu_r.clone()
            spu_cs[:, :, 1:] = torch.where(spu_cs[:, :, 1:] < thresh,
                                             torch.tensor(-5.0), spu_cs[:, :, 1:])
            spu_cs[:, :, 1:] = spu_cs[:, :, 1:] * scale_f
            res = eval_transform(f"5_clip{thresh}_scale{scale_f}",
                                 spu_cs.reshape(T, C), gpu, seqdist, ref, DEVICE)
            results.append(res)

    # --- 6: Affine per-timestep: normalize SPU to match GPU's mean/std per timestep ---
    spu_tn = spu_r.clone()
    for ch in range(1, n_alphabet):
        g_ch = gpu_r[:, :, ch]  # (T, 1024)
        s_ch = spu_r[:, :, ch]  # (T, 1024)
        s_mean = s_ch.mean(dim=1, keepdim=True)
        s_std = s_ch.std(dim=1, keepdim=True).clamp(min=1e-6)
        g_mean = g_ch.mean(dim=1, keepdim=True)
        g_std = g_ch.std(dim=1, keepdim=True).clamp(min=1e-6)
        spu_tn[:, :, ch] = (s_ch - s_mean) / s_std * g_std + g_mean
    res = eval_transform("6_per_timestep_per_ch_normalize",
                         spu_tn.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- 7: Clamp 非 blank 到 [-5, 5] (和 GPU 一样的范围) ---
    spu_clamp = spu_r.clone()
    spu_clamp[:, :, 1:] = spu_clamp[:, :, 1:].clamp(-5.0, 5.0)
    res = eval_transform("7_clamp_-5_to_5",
                         spu_clamp.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- 8: 分段线性: 负值区域更陡 (拉低 SPU 中被抬高的负值) ---
    for neg_scale in [1.3, 1.5, 1.8, 2.0]:
        spu_pw = spu_r.clone()
        nonblank = spu_pw[:, :, 1:]
        neg_mask = nonblank < 0
        nonblank[neg_mask] = nonblank[neg_mask] * neg_scale
        nonblank = nonblank.clamp(-5.0, 5.0)
        spu_pw[:, :, 1:] = nonblank
        res = eval_transform(f"8_neg_scale_{neg_scale}_clamp",
                             spu_pw.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- 9: Quantile matching (match SPU distribution to GPU distribution) ---
    spu_qm = spu_r.clone()
    for ch in range(1, n_alphabet):
        g_ch = gpu_r[:, :, ch].reshape(-1).numpy()
        s_ch = spu_r[:, :, ch].reshape(-1).numpy()
        # Sort SPU, replace with corresponding quantile from GPU
        s_sorted_idx = np.argsort(s_ch)
        g_sorted = np.sort(g_ch)
        matched = np.empty_like(s_ch)
        matched[s_sorted_idx] = g_sorted
        spu_qm[:, :, ch] = torch.from_numpy(matched.reshape(T, n_states))
    res = eval_transform("9_quantile_matching",
                         spu_qm.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # =========================================================
    # 3. 汇总结果
    # =========================================================
    print(f"\n{'='*80}")
    print(f"变换结果汇总  (GPU baseline acc={gpu_acc:.4f}%)")
    print(f"{'='*80}")
    print(f"  {'Method':<42s} {'MAE':>10s} {'MaxAE':>10s} {'CosSim':>10s} {'Acc%':>10s} {'SeqLen':>7s}")
    print(f"  {'-'*89}")

    # Sort by accuracy descending
    results_sorted = sorted(results, key=lambda r: r['acc'], reverse=True)
    for r in results_sorted:
        marker = " ★" if r['acc'] >= gpu_acc else ""
        print(f"  {r['name']:<42s} {r['mae']:>10.4f} {r['max_ae']:>10.4f} "
              f"{r['cos']:>10.6f} {r['acc']:>9.4f}% {r['seq_len']:>6d}{marker}")

    print(f"\n  GPU baseline: acc={gpu_acc:.4f}%, seq_len={len(gpu_seq)}")
    print(f"  ★ = matches or exceeds GPU accuracy")

    # =========================================================
    # 4. 最优变换的参数
    # =========================================================
    best = results_sorted[0]
    print(f"\n{'='*80}")
    print(f"最优变换: {best['name']}")
    print(f"  Accuracy: {best['acc']:.4f}% (vs GPU {gpu_acc:.4f}%, vs original SPU {results[0]['acc']:.4f}%)")
    print(f"  MAE: {best['mae']:.4f} (vs original {results[0]['mae']:.4f})")
    print(f"  CosSim: {best['cos']:.6f} (vs original {results[0]['cos']:.6f})")
    print(f"{'='*80}")
    print("Done.")


if __name__ == "__main__":
    main()
