"""
Sample 139 v2: 更深入的变换实验。

上一轮发现: 全局线性/clip/scale 变换对 Viterbi 解码结果毫无影响 (acc 不变)。
原因: Viterbi 只关注每个 state 内的 argmax 排序，全局变换不改变排序。

本轮重点: 分析排序差异，尝试能改变局部排序的变换。
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


def decode_single(seqdist, scores_single, device):
    scores = scores_single.to(device)
    with torch.no_grad():
        paths = seqdist.viterbi_guided_bidirectional_reshape(
            scores.to(torch.bfloat16), use_bfloat16=True
        )
    return seqdist.path_to_str(paths[0].cpu().numpy())


def eval_transform(name, spu_t, gpu, seqdist, ref, device,
                   n_states=1024, n_alphabet=5):
    T, C = gpu.shape
    diff = (gpu.float() - spu_t.float()).abs()
    mae = diff.mean().item()
    cos = torch.nn.functional.cosine_similarity(
        gpu.float().reshape(1, -1), spu_t.float().reshape(1, -1)).item()

    # Argmax match rate (per state per timestep)
    g_r = gpu.float().reshape(T, n_states, n_alphabet)
    s_r = spu_t.float().reshape(T, n_states, n_alphabet)
    g_argmax = g_r.argmax(dim=2)  # (T, 1024)
    s_argmax = s_r.argmax(dim=2)
    argmax_match = (g_argmax == s_argmax).float().mean().item()

    # Decode
    seq = decode_single(seqdist, spu_t.unsqueeze(1), device)
    acc = accuracy(ref, seq, min_coverage=0.95) if len(seq) else 0.0

    return {"name": name, "mae": mae, "cos": cos, "acc": acc,
            "seq_len": len(seq), "argmax_match": argmax_match}


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

    print("[Loading scores] ...")
    gpu_all = torch.load(FILE_GPU, map_location="cpu")
    spu_all = torch.load(FILE_SPU, map_location="cpu")

    gpu = gpu_all[:, SAMPLE, :].float()
    spu = spu_all[:, SAMPLE, :].float()
    T, C = gpu.shape
    n_states, n_alphabet = 1024, 5

    gpu_r = gpu.reshape(T, n_states, n_alphabet)
    spu_r = spu.reshape(T, n_states, n_alphabet)

    # =========================================================
    # 分析: argmax 排序差异
    # =========================================================
    g_argmax = gpu_r.argmax(dim=2)  # (T, 1024)
    s_argmax = spu_r.argmax(dim=2)
    argmax_match = (g_argmax == s_argmax).float()

    print(f"\n{'='*75}")
    print(f"Argmax 排序分析 (每个 timestep × state 的 argmax(N/A/C/G/T))")
    print(f"{'='*75}")
    total_cells = T * n_states
    match_cnt = argmax_match.sum().item()
    print(f"  Argmax 匹配: {int(match_cnt)}/{total_cells} ({match_cnt/total_cells*100:.2f}%)")
    mismatch_cnt = total_cells - match_cnt
    print(f"  Argmax 不匹配: {int(mismatch_cnt)}/{total_cells} ({mismatch_cnt/total_cells*100:.2f}%)")

    # 不匹配时，GPU 的 argmax 是什么 vs SPU 的 argmax
    mismatch_mask = (g_argmax != s_argmax)
    g_mis = g_argmax[mismatch_mask].numpy()
    s_mis = s_argmax[mismatch_mask].numpy()
    print(f"\n  Argmax 不匹配的分布:")
    print(f"  {'GPU_argmax':>12s} → {'SPU_argmax':>12s}: count")
    for g_val in range(n_alphabet):
        for s_val in range(n_alphabet):
            if g_val == s_val:
                continue
            cnt = ((g_mis == g_val) & (s_mis == s_val)).sum()
            if cnt > 0:
                print(f"  {ALPHABET[g_val]:>12s} → {ALPHABET[s_val]:>12s}: {cnt:>6d}")

    # 不匹配时，GPU argmax 的 score 和 SPU argmax 的 score 差距多大
    mismatch_positions = mismatch_mask.nonzero(as_tuple=False)  # (K, 2): t, state
    if len(mismatch_positions) > 0:
        gpu_winner_scores = []
        spu_winner_scores = []
        margins = []
        for k in range(len(mismatch_positions)):
            t, st = mismatch_positions[k].tolist()
            g_best = g_argmax[t, st].item()
            s_best = s_argmax[t, st].item()
            # SPU 在 GPU-best 通道的 score vs SPU-best 通道的 score
            spu_at_gpu_best = spu_r[t, st, g_best].item()
            spu_at_spu_best = spu_r[t, st, s_best].item()
            margin = spu_at_spu_best - spu_at_gpu_best  # SPU 选错了多少
            margins.append(margin)

        margins = np.array(margins)
        print(f"\n  SPU 在不匹配位置的 margin (SPU_best_score - SPU_at_GPU_best):")
        print(f"    mean={margins.mean():.4f}  median={np.median(margins):.4f}  "
              f"max={margins.max():.4f}  min={margins.min():.4f}")
        print(f"    margin < 0.5: {(margins < 0.5).sum()} ({(margins < 0.5).sum()/len(margins)*100:.1f}%)")
        print(f"    margin < 1.0: {(margins < 1.0).sum()} ({(margins < 1.0).sum()/len(margins)*100:.1f}%)")
        print(f"    margin < 2.0: {(margins < 2.0).sum()} ({(margins < 2.0).sum()/len(margins)*100:.1f}%)")

    # 每 timestep 的 argmax 不匹配率
    per_t_mismatch = mismatch_mask.float().mean(dim=1)  # (T,)
    worst_t = per_t_mismatch.argsort(descending=True)[:10]
    print(f"\n  Argmax 不匹配率最高的 10 个 timestep:")
    for t in worst_t:
        t = t.item()
        print(f"    t={t:<5d}: {per_t_mismatch[t].item()*100:.1f}% 不匹配 "
              f"({int(per_t_mismatch[t].item()*n_states)}/{n_states})")

    # =========================================================
    # GPU baseline
    # =========================================================
    gpu_seq = decode_single(seqdist, gpu.unsqueeze(1), DEVICE)
    gpu_acc = accuracy(ref, gpu_seq, min_coverage=0.95) if len(gpu_seq) else 0.0
    print(f"\n[GPU baseline] acc={gpu_acc:.4f}%")

    # =========================================================
    # 变换实验
    # =========================================================
    results = []

    # 0: Original
    res = eval_transform("0_original_SPU", spu, gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- A: Temperature sharpening (在每个 state 内做 softmax 再 log，降温使分布更尖) ---
    for temp in [0.3, 0.5, 0.7, 0.8, 0.9]:
        spu_sharp = spu_r.clone()
        # softmax with temperature → log → scale back
        logits = spu_sharp / temp
        log_probs = logits - logits.logsumexp(dim=2, keepdim=True)
        # 还原到原始 score 的量纲 (乘以 temp 回去不行，用 scale 近似)
        spu_out = log_probs * temp  # 保持量纲
        res = eval_transform(f"A_temperature_{temp}",
                             spu_out.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- B: 直接用 GPU 的 argmax 来修正 SPU (oracle: 看上界) ---
    spu_oracle = spu_r.clone()
    for t_idx in range(T):
        for st in range(n_states):
            g_best = g_argmax[t_idx, st].item()
            s_best = s_argmax[t_idx, st].item()
            if g_best != s_best:
                # 把 SPU 在 GPU-best 通道的分数提高到最大
                spu_oracle[t_idx, st, g_best] = spu_oracle[t_idx, st].max() + 0.1
    res = eval_transform("B_oracle_fix_argmax",
                         spu_oracle.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- C: 只修正 margin 很小的不匹配 (实际可行的策略) ---
    for margin_thresh in [0.5, 1.0, 2.0, 3.0]:
        spu_fix = spu_r.clone()
        fixed = 0
        for k in range(len(mismatch_positions)):
            t, st = mismatch_positions[k].tolist()
            g_best = g_argmax[t, st].item()
            s_best = s_argmax[t, st].item()
            spu_at_gpu_best = spu_r[t, st, g_best].item()
            spu_at_spu_best = spu_r[t, st, s_best].item()
            margin = spu_at_spu_best - spu_at_gpu_best
            if margin < margin_thresh:
                # 当 margin 很小时，说明不确定，swap 分数让 GPU 的选择获胜
                spu_fix[t, st, g_best] = spu_at_spu_best + 0.01
                spu_fix[t, st, s_best] = spu_at_gpu_best
                fixed += 1
        res = eval_transform(f"C_fix_margin<{margin_thresh}(n={fixed})",
                             spu_fix.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- D: 增强 blank 通道 (让 blank 更dominant，减少假碱基) ---
    for boost in [0.5, 1.0, 1.5, 2.0]:
        spu_bb = spu_r.clone()
        spu_bb[:, :, 0] = spu_bb[:, :, 0] + boost
        res = eval_transform(f"D_boost_blank_+{boost}",
                             spu_bb.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- E: 抑制非 blank 通道 (让弱信号更弱) ---
    for suppress in [0.5, 1.0, 1.5]:
        spu_sup = spu_r.clone()
        nonblank = spu_sup[:, :, 1:]
        spu_sup[:, :, 1:] = nonblank - suppress
        res = eval_transform(f"E_suppress_nonblank_-{suppress}",
                             spu_sup.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- F: 对非blank做 power 变换 (拉大差距) ---
    for power in [1.5, 2.0, 3.0]:
        spu_pow = spu_r.clone()
        nonblank = spu_pow[:, :, 1:]
        sign = nonblank.sign()
        spu_pow[:, :, 1:] = sign * nonblank.abs().pow(1.0/power) * (5.0 ** (1 - 1.0/power))
        spu_pow[:, :, 1:] = spu_pow[:, :, 1:].clamp(-5, 5)
        res = eval_transform(f"F_power_compress_{power}",
                             spu_pow.reshape(T, C), gpu, seqdist, ref, DEVICE)
        results.append(res)

    # --- G: 混合: SPU blank + GPU nonblank (测试非 blank 完全精确时) ---
    spu_mix = spu_r.clone()
    spu_mix[:, :, 1:] = gpu_r[:, :, 1:]
    res = eval_transform("G_oracle_nonblank_from_GPU",
                         spu_mix.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # --- H: Per-timestep regression on non-blank ---
    spu_pt_lr = spu_r.clone()
    for t_idx in range(T):
        for ch in range(1, n_alphabet):
            g_vec = gpu_r[t_idx, :, ch]  # (1024,)
            s_vec = spu_r[t_idx, :, ch]
            # simple affine: match mean & std
            s_mean, s_std = s_vec.mean(), s_vec.std().clamp(min=1e-6)
            g_mean, g_std = g_vec.mean(), g_vec.std().clamp(min=1e-6)
            spu_pt_lr[t_idx, :, ch] = (s_vec - s_mean) / s_std * g_std + g_mean
    res = eval_transform("H_per_t_per_ch_affine",
                         spu_pt_lr.reshape(T, C), gpu, seqdist, ref, DEVICE)
    results.append(res)

    # =========================================================
    # 结果汇总
    # =========================================================
    print(f"\n{'='*90}")
    print(f"变换结果汇总  (GPU baseline acc={gpu_acc:.4f}%)")
    print(f"{'='*90}")
    print(f"  {'Method':<40s} {'MAE':>8s} {'CosSim':>10s} {'ArgmaxMatch':>12s} {'Acc%':>10s} {'Len':>5s}")
    print(f"  {'-'*90}")

    results_sorted = sorted(results, key=lambda r: r['acc'], reverse=True)
    for r in results_sorted:
        marker = " ★" if r['acc'] >= gpu_acc else ""
        print(f"  {r['name']:<40s} {r['mae']:>8.4f} {r['cos']:>10.6f} "
              f"{r['argmax_match']*100:>11.2f}% {r['acc']:>9.4f}% {r['seq_len']:>5d}{marker}")

    print(f"\n  GPU baseline: acc={gpu_acc:.4f}%, seq_len={len(gpu_seq)}")

    print(f"\n{'='*90}")
    print("Done.")


if __name__ == "__main__":
    main()
