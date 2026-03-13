"""
Deep-dive into sample 139: compare GPU vs SPU scores through the full
Viterbi decode pipeline, showing where the paths diverge.
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)

from viterbi_0224 import (
    CTC_CRF,
    decode_ref,
    accuracy,
    parasail_to_sam,
    TrainingDataSet3,
)

# ---- Paths ----
FILE_GPU = os.path.join(SCRIPT_DIR, "scores_gpu.pt")
FILE_SPU = os.path.join(SCRIPT_DIR, "spu_scores_512.pt")
CONFIG_PATH = "/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/config.toml"
DATA_DIR = "/workspace/huada/moffett_data/250F600274011_train_data/val"
DEVICE = "cuda:0"
SAMPLE_IDX = 139


def viterbi_debug(seqdist, scores_single, device):
    """
    Run Viterbi on a single sample (T, 1, C) and return detailed intermediates.
    """
    scores = scores_single.to(device)
    T, N, _ = scores.shape
    assert N == 1

    n_states = seqdist.n_base ** seqdist.state_len
    n_alphabet = len(seqdist.alphabet)
    dtype = torch.bfloat16

    # Build idx tables
    idx_np = np.zeros((n_states, n_alphabet), dtype=np.int32)
    idx_np[:, 0] = np.arange(n_states)
    for j in range(1, n_alphabet):
        idx_np[:, j] = ((j - 1) * n_states + np.arange(n_states)) // seqdist.n_base
    idx = torch.from_numpy(idx_np).to(device=device, dtype=torch.long)

    idx_T_np = np.zeros((n_states, n_alphabet), dtype=np.int32)
    for i in range(n_states):
        idx_T_np[i][0] = i * n_alphabet
    for i in range(n_states):
        for j in range(seqdist.n_base):
            repeat_idx = i * seqdist.n_base + j
            repeat_idx_row = repeat_idx % n_states
            repeat_idx_col = 1 + repeat_idx // n_states
            flatten_idx = repeat_idx_row * n_alphabet + repeat_idx_col
            idx_T_np[i][j + 1] = flatten_idx
    idx_T = torch.from_numpy(idx_T_np).to(device=device, dtype=torch.long)
    idx_T_targets = idx_T // n_alphabet

    Ms = scores.transpose(1, 2).to(dtype).reshape(T, n_states, n_alphabet, N)
    Ms_flat = Ms.reshape(T, -1, N)
    Ms_T = Ms_flat[:, idx_T, :]

    segment_size = 8

    # Forward
    alphas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype)
    alpha = alphas_all[0]
    for t in range(T):
        alpha_indexed = alpha[idx, :]
        candidates = alpha_indexed + Ms[t]
        alpha = torch.logsumexp(candidates, dim=1)
        if t % segment_size == 0:
            alpha_min = alpha.min(dim=0, keepdim=True)[0]
            alpha = alpha - alpha_min
        alphas_all[t + 1] = alpha

    # Backward
    betas_all = torch.zeros(T + 1, n_states, N, device=device, dtype=dtype)
    beta = betas_all[T]
    for t in range(T - 1, -1, -1):
        beta_indexed = beta[idx_T_targets, :]
        candidates = Ms_T[t] + beta_indexed
        beta = torch.logsumexp(candidates, dim=1)
        if t % segment_size == 0:
            beta_min = beta.min(dim=0, keepdim=True)[0]
            beta = beta - beta_min
        betas_all[t] = beta

    # Viterbi forward with guided scores + record intermediates
    alpha_max = torch.full((n_states, N), float('-inf'), device=device, dtype=dtype)
    alpha_max[0, :] = 0.0
    traceback = torch.zeros(T, n_states, N, dtype=torch.int8, device=device)
    best_state_per_t = []
    best_edge_per_t = []
    guided_max_per_t = []

    for t in range(T):
        alpha_indexed_src = alphas_all[t][idx, :]
        beta_indexed_dst = betas_all[t + 1][:, None, :]
        guided_scores_t = alpha_indexed_src + Ms[t] + beta_indexed_dst

        alpha_indexed_max = alpha_max[idx, :]
        candidates = alpha_indexed_max + guided_scores_t
        alpha_max, best_z = candidates.max(dim=1)
        traceback[t] = best_z.to(torch.int8)

        # Record the best state at this timestep
        best_state_per_t.append(alpha_max.argmax(dim=0).item())
        best_edge_per_t.append(best_z[alpha_max.argmax(dim=0).item(), 0].item())
        guided_max_per_t.append(alpha_max.max().item())

        if t % 8 == 0:
            alpha_max = alpha_max - alpha_max.max(dim=0, keepdim=True)[0]

    # Traceback
    current_states = alpha_max.argmax(dim=0)
    paths = torch.zeros(T, N, dtype=torch.int8, device=device)
    batch_idx = torch.arange(N, device=device)

    path_states = []
    for t in range(T - 1, -1, -1):
        best_edges = traceback[t, current_states, batch_idx]
        paths[t] = best_edges
        path_states.append(current_states.item())
        current_states = idx[current_states, best_edges.long()]
    path_states.reverse()

    path = paths[:, 0].cpu().numpy()
    seq = seqdist.path_to_str(path)

    return {
        "seq": seq,
        "path": path,
        "path_states": np.array(path_states),
        "best_state_per_t": np.array(best_state_per_t),
    }


def main():
    torch.manual_seed(25)
    torch.cuda.manual_seed_all(25)
    np.random.seed(25)
    random.seed(25)

    config = toml.load(CONFIG_PATH)
    state_len = config["global_norm"]["state_len"]
    alphabet = config["labels"]["labels"]
    seqdist = CTC_CRF(state_len=state_len, alphabet=alphabet)

    # ---- Load target for sample 139 ----
    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(DATA_DIR, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    for data, target, *_ in loader:
        targets = list(torch.unbind(target, 0))
        break
    ref = decode_ref(targets[SAMPLE_IDX], alphabet)

    # ---- Load scores ----
    print(f"[Loading scores] ...")
    scores_gpu_all = torch.load(FILE_GPU, map_location="cpu")
    scores_spu_all = torch.load(FILE_SPU, map_location="cpu")

    # Extract sample 139: (T, C) → (T, 1, C)
    s_gpu = scores_gpu_all[:, SAMPLE_IDX:SAMPLE_IDX+1, :].contiguous()
    s_spu = scores_spu_all[:, SAMPLE_IDX:SAMPLE_IDX+1, :].contiguous()
    T, _, C = s_gpu.shape

    print(f"\n{'='*75}")
    print(f"Sample {SAMPLE_IDX} Deep Dive")
    print(f"{'='*75}")
    print(f"  Reference length: {len(ref)}")
    print(f"  Reference (first 100 chars): {ref[:100]}...")
    print(f"  Score shape: ({T}, 1, {C})")

    # ---- Score-level comparison ----
    diff = (s_gpu.float() - s_spu.float()).abs()
    print(f"\n--- Score Comparison (sample {SAMPLE_IDX}) ---")
    print(f"  MAE:     {diff.mean().item():.6e}")
    print(f"  Max AE:  {diff.max().item():.6e}")
    print(f"  RMSE:    {(s_gpu.float() - s_spu.float()).pow(2).mean().sqrt().item():.6e}")

    # Per-timestep MAE
    per_t_mae = diff.mean(dim=(1, 2)).numpy()
    worst_ts = np.argsort(-per_t_mae)[:10]
    print(f"\n  Top 10 worst timesteps (by MAE):")
    for t in worst_ts:
        print(f"    t={t:<5d} MAE={per_t_mae[t]:.6e}")

    # ---- Decode both with debug ----
    print(f"\n--- Viterbi Decode (GPU scores) ---")
    res_gpu = viterbi_debug(seqdist, s_gpu, DEVICE)

    print(f"--- Viterbi Decode (SPU scores) ---")
    res_spu = viterbi_debug(seqdist, s_spu, DEVICE)

    print(f"\n--- Decoded Sequences ---")
    print(f"  GPU seq length: {len(res_gpu['seq'])}")
    print(f"  SPU seq length: {len(res_spu['seq'])}")
    print(f"  REF seq length: {len(ref)}")

    print(f"\n  GPU seq (first 200): {res_gpu['seq'][:200]}")
    print(f"  SPU seq (first 200): {res_spu['seq'][:200]}")
    print(f"  REF     (first 200): {ref[:200]}")

    # ---- Accuracy ----
    split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")
    acc_gpu = accuracy(ref, res_gpu['seq'], min_coverage=0.95) if len(res_gpu['seq']) else 0.0
    acc_spu = accuracy(ref, res_spu['seq'], min_coverage=0.95) if len(res_spu['seq']) else 0.0
    print(f"\n--- Accuracy ---")
    print(f"  GPU: {acc_gpu:.4f}%")
    print(f"  SPU: {acc_spu:.4f}%")
    print(f"  diff: {acc_spu - acc_gpu:+.4f}%")

    # ---- Path comparison ----
    path_gpu = res_gpu['path']  # (T,) edges: 0=blank, 1-4=ACGT
    path_spu = res_spu['path']
    states_gpu = res_gpu['path_states']
    states_spu = res_spu['path_states']

    path_match = (path_gpu == path_spu)
    state_match = (states_gpu == states_spu)
    print(f"\n--- Path Comparison ---")
    print(f"  Edge match:  {path_match.sum()}/{T} ({path_match.sum()/T*100:.2f}%)")
    print(f"  State match: {state_match.sum()}/{T} ({state_match.sum()/T*100:.2f}%)")

    # Find first divergence
    diff_positions = np.where(~path_match)[0]
    if len(diff_positions) > 0:
        first_diff = diff_positions[0]
        print(f"\n  First path divergence at t={first_diff}")
        # Show context around first divergence
        start = max(0, first_diff - 5)
        end = min(T, first_diff + 15)
        edge_labels = ['N', 'A', 'C', 'G', 'T']
        print(f"\n  {'t':>6s}  {'GPU_edge':>10s} {'SPU_edge':>10s} {'GPU_state':>10s} {'SPU_state':>10s} {'score_MAE':>12s} {'match':>6s}")
        print(f"  {'-'*62}")
        for t in range(start, end):
            marker = "  " if path_match[t] else " *"
            print(f"  {t:>6d}  {edge_labels[path_gpu[t]]:>10s} {edge_labels[path_spu[t]]:>10s} "
                  f"{states_gpu[t]:>10d} {states_spu[t]:>10d} "
                  f"{per_t_mae[t]:>12.6e} {marker}")

        # Show all divergence regions (contiguous blocks)
        print(f"\n  --- Divergence regions (contiguous blocks) ---")
        block_starts = [diff_positions[0]]
        block_ends = []
        for k in range(1, len(diff_positions)):
            if diff_positions[k] != diff_positions[k-1] + 1:
                block_ends.append(diff_positions[k-1])
                block_starts.append(diff_positions[k])
        block_ends.append(diff_positions[-1])

        print(f"  Total divergence blocks: {len(block_starts)}")
        print(f"  {'Block':>6s}  {'Start':>6s}  {'End':>6s}  {'Length':>6s}")
        for bi in range(min(30, len(block_starts))):
            blen = block_ends[bi] - block_starts[bi] + 1
            print(f"  {bi+1:>6d}  {block_starts[bi]:>6d}  {block_ends[bi]:>6d}  {blen:>6d}")
        if len(block_starts) > 30:
            print(f"  ... ({len(block_starts) - 30} more blocks)")

    else:
        print(f"\n  Paths are IDENTICAL!")

    # ---- Emission diff at divergence points ----
    if len(diff_positions) > 0:
        print(f"\n--- Score details at first 5 divergence points ---")
        n_states = seqdist.n_base ** seqdist.state_len
        n_alphabet = len(seqdist.alphabet)
        for di in range(min(5, len(diff_positions))):
            t = diff_positions[di]
            st_gpu = states_gpu[t]
            st_spu = states_spu[t]
            # Show the 5 emission scores at that state for both GPU and SPU
            gpu_scores_t = s_gpu[t, 0, :].float().reshape(n_states, n_alphabet)
            spu_scores_t = s_spu[t, 0, :].float().reshape(n_states, n_alphabet)

            print(f"\n  t={t}, GPU_state={st_gpu}, SPU_state={st_spu}")
            print(f"    GPU emissions at state {st_gpu}: {gpu_scores_t[st_gpu].numpy()}")
            print(f"    SPU emissions at state {st_gpu}: {spu_scores_t[st_gpu].numpy()}")
            print(f"    diff:                            {(gpu_scores_t[st_gpu] - spu_scores_t[st_gpu]).numpy()}")
            if st_spu != st_gpu:
                print(f"    GPU emissions at state {st_spu}: {gpu_scores_t[st_spu].numpy()}")
                print(f"    SPU emissions at state {st_spu}: {spu_scores_t[st_spu].numpy()}")
                print(f"    diff:                            {(gpu_scores_t[st_spu] - spu_scores_t[st_spu]).numpy()}")

    print(f"\n{'='*75}")
    print("Done.")


if __name__ == "__main__":
    main()
