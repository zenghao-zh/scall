"""
Sample 139: SPU 自修复实验 — 只用 SPU 自身的数据做变换，不参考 GPU。

分析思路:
  - Blank 通道完全准确 (MAE=0)
  - 73.8% 的 argmax 错误是: GPU 选碱基，SPU 却选了 blank
  - 原因: SPU 的非 blank 值被压缩 (正值偏低, 负值偏高), 导致很多碱基信号不够强
  - 因此: 需要拉大非 blank 通道的动态范围，让强信号更强、弱信号更弱
"""

import os, sys, random
import torch
import numpy as np

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
ALPHABET = ['N', 'A', 'C', 'G', 'T']
N_STATES, N_ALPHA = 1024, 5


def decode_single(seqdist, scores_2d, device):
    """scores_2d: (T, C) → unsqueeze → (T,1,C) → decode"""
    s = scores_2d.unsqueeze(1).to(device)
    with torch.no_grad():
        paths = seqdist.viterbi_guided_bidirectional_reshape(
            s.to(torch.bfloat16), use_bfloat16=True)
    return seqdist.path_to_str(paths[0].cpu().numpy())


def eval_tx(name, spu_2d, gpu_2d, seqdist, ref, device):
    T, C = gpu_2d.shape
    diff = (gpu_2d.float() - spu_2d.float()).abs()
    mae = diff.mean().item()

    g_r = gpu_2d.float().reshape(T, N_STATES, N_ALPHA)
    s_r = spu_2d.float().reshape(T, N_STATES, N_ALPHA)
    argmax_match = (g_r.argmax(2) == s_r.argmax(2)).float().mean().item()

    seq = decode_single(seqdist, spu_2d, device)
    acc = accuracy(ref, seq, min_coverage=0.95) if len(seq) else 0.0
    return {"name": name, "mae": mae, "acc": acc, "len": len(seq),
            "argmax": argmax_match}


def main():
    torch.manual_seed(25); torch.cuda.manual_seed_all(25)
    np.random.seed(25); random.seed(25)

    config = toml.load(CONFIG_PATH)
    seqdist = CTC_CRF(state_len=config["global_norm"]["state_len"],
                       alphabet=config["labels"]["labels"])

    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(DATA_DIR, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    for data, target, *_ in loader:
        targets = list(torch.unbind(target, 0))
        break
    ref = decode_ref(targets[SAMPLE], config["labels"]["labels"])

    print("[Loading] ...")
    gpu_all = torch.load(FILE_GPU, map_location="cpu")
    spu_all = torch.load(FILE_SPU, map_location="cpu")
    gpu = gpu_all[:, SAMPLE, :].float()
    spu = spu_all[:, SAMPLE, :].float()
    T, C = gpu.shape

    spu_r = spu.reshape(T, N_STATES, N_ALPHA)
    gpu_r = gpu.reshape(T, N_STATES, N_ALPHA)

    gpu_seq = decode_single(seqdist, gpu, DEVICE)
    gpu_acc = accuracy(ref, gpu_seq, min_coverage=0.95) if len(gpu_seq) else 0.0
    print(f"[GPU baseline] acc={gpu_acc:.4f}%, len={len(gpu_seq)}")
    print(f"[Reference] len={len(ref)}")

    # =========================================================
    # 先分析 SPU 自身的数值特征
    # =========================================================
    blank = spu_r[:, :, 0]       # (T, 1024) — blank 通道
    nonblank = spu_r[:, :, 1:]   # (T, 1024, 4) — 碱基通道

    # 在每个 (t, state) 位置: blank score vs 最大碱基 score
    max_base, max_base_idx = nonblank.max(dim=2)  # (T, 1024)
    base_minus_blank = max_base - blank  # >0 说明碱基赢, <0 说明 blank 赢

    print(f"\n{'='*75}")
    print(f"SPU 自身特征分析")
    print(f"{'='*75}")
    print(f"  Blank 值: 全部 = {blank[0,0].item():.4f} (常数)")
    print(f"  非 blank 值范围: [{nonblank.min().item():.4f}, {nonblank.max().item():.4f}]")
    print(f"  非 blank 均值: {nonblank.mean().item():.4f}")
    print(f"  非 blank std: {nonblank.std().item():.4f}")
    print(f"\n  max_base - blank 分布:")
    print(f"    mean={base_minus_blank.mean().item():.4f}")
    print(f"    >0 (碱基赢): {(base_minus_blank > 0).sum().item()}/{base_minus_blank.numel()} "
          f"({(base_minus_blank > 0).float().mean().item()*100:.1f}%)")

    # 对比 GPU 的同一指标
    g_blank = gpu_r[:, :, 0]
    g_nonblank = gpu_r[:, :, 1:]
    g_max_base, _ = g_nonblank.max(dim=2)
    g_bmb = g_max_base - g_blank
    print(f"\n  [对比] GPU 的 max_base - blank:")
    print(f"    mean={g_bmb.mean().item():.4f}")
    print(f"    >0 (碱基赢): {(g_bmb > 0).sum().item()}/{g_bmb.numel()} "
          f"({(g_bmb > 0).float().mean().item()*100:.1f}%)")

    # 那些 GPU 碱基赢但 SPU blank 赢的位置
    flip_to_blank = (g_bmb > 0) & (base_minus_blank <= 0)
    flip_to_base = (g_bmb <= 0) & (base_minus_blank > 0)
    print(f"\n  GPU碱基赢 → SPU blank赢 (翻转): {flip_to_blank.sum().item()}")
    print(f"  GPU blank赢 → SPU碱基赢 (翻转): {flip_to_base.sum().item()}")

    # =========================================================
    # 实验: 只用 SPU 自己的信息做变换
    # =========================================================
    results = []

    # 0: baseline
    res = eval_tx("0_original_SPU", spu, gpu, seqdist, ref, DEVICE)
    results.append(res)

    # ----- 策略 A: 非 blank 通道整体加偏移 (让碱基更容易赢过 blank) -----
    for bias in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        s = spu_r.clone()
        s[:, :, 1:] = s[:, :, 1:] + bias
        results.append(eval_tx(f"A_nonblank+{bias}", s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 B: blank 通道减偏移 -----
    for bias in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        s = spu_r.clone()
        s[:, :, 0] = s[:, :, 0] - bias
        results.append(eval_tx(f"B_blank-{bias}", s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 C: 非 blank 通道以 blank 为中心做拉伸 -----
    #   new_score = blank + (score - blank) * scale
    for scale in [1.1, 1.2, 1.3, 1.5, 1.8, 2.0]:
        s = spu_r.clone()
        for ch in range(1, N_ALPHA):
            s[:, :, ch] = blank + (s[:, :, ch] - blank) * scale
        results.append(eval_tx(f"C_stretch_from_blank_x{scale}",
                               s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 D: 每个 state 内做 softmax 降温再映射回 score -----
    #   降温让 argmax 更确定，但保持相对量纲
    for temp in [0.5, 0.7, 0.8, 0.9]:
        s = spu_r.clone()
        logits = s / temp
        log_sm = logits - logits.logsumexp(dim=2, keepdim=True)
        # 映射回原始量纲: 保持 max 值不变，仅改变分布形状
        orig_max = s.max(dim=2, keepdim=True).values
        new_max = log_sm.max(dim=2, keepdim=True).values
        s_out = log_sm - new_max + orig_max
        results.append(eval_tx(f"D_sharpen_temp{temp}_keepmax",
                               s_out.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 E: 非 blank 通道以自身均值为中心拉伸 -----
    for scale in [1.1, 1.2, 1.3, 1.5, 2.0]:
        s = spu_r.clone()
        nb = s[:, :, 1:]  # (T, 1024, 4)
        # per (t, state): mean of 4 base scores
        nb_mean = nb.mean(dim=2, keepdim=True)
        s[:, :, 1:] = nb_mean + (nb - nb_mean) * scale
        results.append(eval_tx(f"E_stretch_nonblank_mean_x{scale}",
                               s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 F: 把非 blank 中低于某百分位的值压到 -5 -----
    nb_flat = nonblank.reshape(-1)
    for pct in [10, 20, 30, 40, 50]:
        thresh = torch.quantile(nb_flat, pct / 100.0).item()
        s = spu_r.clone()
        mask = s[:, :, 1:] < thresh
        s[:, :, 1:] = torch.where(mask, torch.tensor(-5.0), s[:, :, 1:])
        results.append(eval_tx(f"F_bottom_{pct}pct_to_-5(thresh={thresh:.2f})",
                               s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 G: 每个 state 内只保留最大的碱基，其余压到 -5 -----
    s = spu_r.clone()
    nb = s[:, :, 1:]  # (T, 1024, 4)
    max_idx = nb.argmax(dim=2, keepdim=True)
    mask = torch.zeros_like(nb, dtype=torch.bool)
    mask.scatter_(2, max_idx, True)
    nb[~mask] = -5.0
    s[:, :, 1:] = nb
    results.append(eval_tx("G_keep_top1_base_only",
                           s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 H: 组合 — 非 blank + offset & 拉伸 -----
    for bias in [0.2, 0.3, 0.5]:
        for scale in [1.2, 1.3, 1.5]:
            s = spu_r.clone()
            s[:, :, 1:] = s[:, :, 1:] + bias
            for ch in range(1, N_ALPHA):
                s[:, :, ch] = blank + (s[:, :, ch] - blank) * scale
            results.append(eval_tx(f"H_bias+{bias}_stretch{scale}",
                                   s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 I: 每 timestep 对非 blank 做 z-score 标准化再映射到固定范围 -----
    for target_std in [1.5, 2.0, 2.5, 3.0]:
        s = spu_r.clone()
        nb = s[:, :, 1:].reshape(T, -1)  # (T, 4096)
        nb_mean = nb.mean(dim=1, keepdim=True)
        nb_std = nb.std(dim=1, keepdim=True).clamp(min=1e-6)
        nb_norm = (nb - nb_mean) / nb_std * target_std + nb_mean
        s[:, :, 1:] = nb_norm.reshape(T, N_STATES, N_ALPHA - 1)
        results.append(eval_tx(f"I_per_t_normalize_std={target_std}",
                               s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # ----- 策略 J: 在碱基信号强的位置 (max_base > blank+margin) 额外 boost -----
    for margin in [-0.5, 0.0, 0.5]:
        for boost in [0.3, 0.5, 1.0]:
            s = spu_r.clone()
            nb = s[:, :, 1:]
            mb, _ = nb.max(dim=2, keepdim=True)
            strong_mask = (mb > blank.unsqueeze(2) + margin)  # (T, 1024, 1)
            strong_mask = strong_mask.expand_as(nb)
            nb[strong_mask] = nb[strong_mask] + boost
            s[:, :, 1:] = nb
            results.append(eval_tx(f"J_boost_strong(m={margin},b=+{boost})",
                                   s.reshape(T,C), gpu, seqdist, ref, DEVICE))

    # =========================================================
    # 结果汇总
    # =========================================================
    print(f"\n{'='*85}")
    print(f"SPU 自修复实验结果  (GPU baseline acc={gpu_acc:.4f}%)")
    print(f"{'='*85}")
    print(f"  {'Method':<45s} {'MAE':>8s} {'Argmax%':>8s} {'Acc%':>10s} {'Len':>5s}")
    print(f"  {'-'*80}")

    results_sorted = sorted(results, key=lambda r: r['acc'], reverse=True)
    for r in results_sorted:
        marker = " ★" if r['acc'] >= gpu_acc else ""
        print(f"  {r['name']:<45s} {r['mae']:>8.4f} {r['argmax']*100:>7.2f}% "
              f"{r['acc']:>9.4f}% {r['len']:>5d}{marker}")

    # 最优
    best = results_sorted[0]
    base = results[0]
    print(f"\n  最优: {best['name']}")
    print(f"    Accuracy: {best['acc']:.4f}% (原始SPU: {base['acc']:.4f}%, GPU: {gpu_acc:.4f}%)")
    print(f"    提升: {best['acc'] - base['acc']:+.4f}%")

    print(f"\n{'='*85}")
    print("Done.")


if __name__ == "__main__":
    main()
