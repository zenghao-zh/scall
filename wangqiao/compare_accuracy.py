"""
Compare decoding accuracy between scores_gpu.pt and spu_scores_512.pt.

Loads both score tensors, runs Viterbi decode on each, computes per-sample
accuracy against the validation targets, and reports the difference.
"""

import os
import sys
import re
import time
import random
from collections import defaultdict

import toml
import torch
import numpy as np
import parasail

# ---- Reuse components from viterbi_0224.py ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from viterbi_0224 import (
    CTC_CRF,
    decode_ref,
    accuracy,
    TrainingDataSet3,
)

# ---- Paths ----
FILE_GPU = os.path.join(SCRIPT_DIR, "scores_gpu.pt")
FILE_SPU = os.path.join(SCRIPT_DIR, "spu_scores_512.pt")
CONFIG_PATH = "/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/config.toml"
DATA_DIR = "/workspace/huada/moffett_data/250F600274011_train_data/val"
DEVICE = "cuda:0"


def decode_scores(seqdist, scores, device):
    """Run Viterbi decode on a scores tensor (T, N, C) → list of strings."""
    scores = scores.to(device)
    with torch.no_grad():
        paths = seqdist.viterbi_guided_bidirectional_reshape(
            scores.to(torch.bfloat16), use_bfloat16=True
        )
    return [seqdist.path_to_str(path) for path in paths.cpu().numpy()]


def main():
    torch.manual_seed(25)
    torch.cuda.manual_seed_all(25)
    np.random.seed(25)
    random.seed(25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Load config ----
    config = toml.load(CONFIG_PATH)
    state_len = config["global_norm"]["state_len"]
    alphabet = config["labels"]["labels"]
    print(f"[Config] state_len={state_len}, alphabet={alphabet}")

    # ---- Build decoder ----
    seqdist = CTC_CRF(state_len=state_len, alphabet=alphabet)

    # ---- Load validation targets ----
    print(f"[Loading validation data] {DATA_DIR}")
    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(DATA_DIR, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=True)
    # Get the first batch targets (matching the 512 batch)
    for data, target, *_ in loader:
        targets = list(torch.unbind(target, 0))
        break
    refs = [decode_ref(t, alphabet) for t in targets]
    N = len(refs)
    print(f"[Targets] {N} reference sequences loaded")

    # ---- Load scores ----
    print(f"\n[Loading scores_gpu.pt] ...")
    scores_gpu = torch.load(FILE_GPU, map_location="cpu")
    print(f"  shape={scores_gpu.shape}, dtype={scores_gpu.dtype}")

    print(f"[Loading spu_scores_512.pt] ...")
    scores_spu = torch.load(FILE_SPU, map_location="cpu")
    print(f"  shape={scores_spu.shape}, dtype={scores_spu.dtype}")

    assert scores_gpu.shape[1] == N, \
        f"Batch size mismatch: scores_gpu has {scores_gpu.shape[1]}, targets has {N}"

    # ---- Decode GPU scores ----
    print(f"\n[Decoding scores_gpu (Viterbi)] ...")
    t0 = time.perf_counter()
    seqs_gpu = decode_scores(seqdist, scores_gpu, DEVICE)
    t_gpu = time.perf_counter() - t0
    print(f"  Done in {t_gpu:.2f}s")

    # ---- Decode SPU scores ----
    print(f"[Decoding spu_scores_512 (Viterbi)] ...")
    t0 = time.perf_counter()
    seqs_spu = decode_scores(seqdist, scores_spu, DEVICE)
    t_spu = time.perf_counter() - t0
    print(f"  Done in {t_spu:.2f}s")

    # ---- Compute per-sample accuracy ----
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=0.95)

    acc_gpu = []
    acc_spu = []
    for i in range(N):
        a_gpu = accuracy_with_cov(refs[i], seqs_gpu[i]) if len(seqs_gpu[i]) else 0.0
        a_spu = accuracy_with_cov(refs[i], seqs_spu[i]) if len(seqs_spu[i]) else 0.0
        acc_gpu.append(a_gpu)
        acc_spu.append(a_spu)

    acc_gpu = np.array(acc_gpu)
    acc_spu = np.array(acc_spu)
    acc_diff = acc_spu - acc_gpu  # positive = SPU better, negative = GPU better

    # ---- Overall stats ----
    print(f"\n{'='*75}")
    print(f"Overall Accuracy Comparison  (N={N} samples)")
    print(f"{'='*75}")
    print(f"  {'':30s} {'scores_gpu':>14s} {'spu_scores_512':>14s} {'diff':>10s}")
    print(f"  {'-'*68}")
    print(f"  {'Mean accuracy':30s} {np.mean(acc_gpu):>13.4f}% {np.mean(acc_spu):>13.4f}% {np.mean(acc_diff):>+9.4f}%")
    print(f"  {'Median accuracy':30s} {np.median(acc_gpu):>13.4f}% {np.median(acc_spu):>13.4f}% {np.median(acc_diff):>+9.4f}%")
    print(f"  {'Std accuracy':30s} {np.std(acc_gpu):>13.4f}% {np.std(acc_spu):>13.4f}%")
    print(f"  {'Min accuracy':30s} {np.min(acc_gpu):>13.4f}% {np.min(acc_spu):>13.4f}%")
    print(f"  {'Max accuracy':30s} {np.max(acc_gpu):>13.4f}% {np.max(acc_spu):>13.4f}%")

    # ---- Count how many samples differ ----
    same = np.sum(acc_diff == 0)
    spu_better = np.sum(acc_diff > 0)
    gpu_better = np.sum(acc_diff < 0)
    print(f"\n  Samples where SPU better:   {spu_better:>4d} / {N}")
    print(f"  Samples where GPU better:   {gpu_better:>4d} / {N}")
    print(f"  Samples with same accuracy: {same:>4d} / {N}")

    # Tolerance-based comparison
    print(f"\n  --- Accuracy difference distribution ---")
    for tol in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]:
        within = np.sum(np.abs(acc_diff) <= tol)
        print(f"  |acc_diff| <= {tol:>5.2f}%: {within:>4d} / {N}  ({within/N*100:.1f}%)")

    # ---- Top worst samples (SPU lost most accuracy vs GPU) ----
    sorted_idx = np.argsort(acc_diff)  # ascending: most negative first (GPU better)
    print(f"\n{'='*75}")
    print(f"Top 20 samples where SPU accuracy DROPS most vs GPU")
    print(f"{'='*75}")
    print(f"  {'Rank':<6} {'Sample':<8} {'GPU acc%':>10} {'SPU acc%':>10} {'diff':>10} {'seq_len_gpu':>12} {'seq_len_spu':>12}")
    print(f"  {'-'*68}")
    for rank in range(min(20, N)):
        i = sorted_idx[rank]
        print(f"  {rank+1:<6} {i:<8} {acc_gpu[i]:>9.4f}% {acc_spu[i]:>9.4f}% {acc_diff[i]:>+9.4f}% "
              f"{len(seqs_gpu[i]):>12d} {len(seqs_spu[i]):>12d}")

    # ---- Top best samples (SPU gained accuracy vs GPU) ----
    print(f"\n{'='*75}")
    print(f"Top 20 samples where SPU accuracy IMPROVES most vs GPU")
    print(f"{'='*75}")
    print(f"  {'Rank':<6} {'Sample':<8} {'GPU acc%':>10} {'SPU acc%':>10} {'diff':>10} {'seq_len_gpu':>12} {'seq_len_spu':>12}")
    print(f"  {'-'*68}")
    for rank in range(min(20, N)):
        i = sorted_idx[N - 1 - rank]
        print(f"  {rank+1:<6} {i:<8} {acc_gpu[i]:>9.4f}% {acc_spu[i]:>9.4f}% {acc_diff[i]:>+9.4f}% "
              f"{len(seqs_gpu[i]):>12d} {len(seqs_spu[i]):>12d}")

    # ---- Cross-reference with score MAE ----
    print(f"\n{'='*75}")
    print(f"Score MAE vs Accuracy Difference (per sample)")
    print(f"{'='*75}")
    abs_diff = (scores_gpu.float() - scores_spu.float()).abs()
    per_sample_mae = abs_diff.mean(dim=(0, 2)).numpy()  # (N,)

    # Sort by MAE descending
    mae_sorted_idx = np.argsort(-per_sample_mae)
    print(f"\n  Top 20 samples by score MAE:")
    print(f"  {'Rank':<6} {'Sample':<8} {'MAE':>12} {'GPU acc%':>10} {'SPU acc%':>10} {'acc_diff':>10}")
    print(f"  {'-'*62}")
    for rank in range(min(20, N)):
        i = mae_sorted_idx[rank]
        print(f"  {rank+1:<6} {i:<8} {per_sample_mae[i]:>12.6f} {acc_gpu[i]:>9.4f}% {acc_spu[i]:>9.4f}% {acc_diff[i]:>+9.4f}%")

    # Correlation between MAE and accuracy drop
    corr = np.corrcoef(per_sample_mae, np.abs(acc_diff))[0, 1]
    print(f"\n  Pearson correlation (MAE vs |acc_diff|): {corr:.6f}")

    print(f"\n{'='*75}")
    print("Done.")


if __name__ == "__main__":
    main()
