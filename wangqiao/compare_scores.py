"""
Compare similarity between scores_gpu.pt and spu_scores_512.pt

Metrics:
  - Shape / dtype
  - Max / Mean absolute error
  - Relative error
  - Cosine similarity
  - Pearson correlation
  - RMSE
  - Histogram of absolute errors
  - Per-position error statistics
"""

import os
import sys
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_A = os.path.join(SCRIPT_DIR, "scores_gpu.pt")
FILE_B = os.path.join(SCRIPT_DIR, "spu_scores_512.pt")


def load_tensor(path):
    print(f"Loading {os.path.basename(path)} ...")
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        print(f"  -> dict with keys: {list(data.keys())}")
        # if it's a state_dict-like, try to concatenate all values
        tensors = {k: v for k, v in data.items() if isinstance(v, torch.Tensor)}
        if len(tensors) == 1:
            key = list(tensors.keys())[0]
            data = tensors[key]
            print(f"  -> Using key '{key}'")
        else:
            print(f"  -> Multiple tensors found, returning dict")
            return data
    elif isinstance(data, (list, tuple)):
        print(f"  -> list/tuple with {len(data)} elements")
        if len(data) == 1:
            data = data[0]
    print(f"  -> shape={data.shape}, dtype={data.dtype}")
    return data


def compare_tensors(a, b, name=""):
    """Compare two tensors and print similarity metrics."""
    prefix = f"[{name}] " if name else ""

    # Convert to float32 for comparison
    a_f = a.float()
    b_f = b.float()

    diff = (a_f - b_f)
    abs_diff = diff.abs()

    # --- Basic stats ---
    max_ae = abs_diff.max().item()
    mean_ae = abs_diff.mean().item()
    median_ae = abs_diff.median().item()
    rmse = diff.pow(2).mean().sqrt().item()

    print(f"\n{'='*70}")
    print(f"{prefix}Comparison Results")
    print(f"{'='*70}")
    print(f"  Shape A:  {tuple(a.shape)},  dtype: {a.dtype}")
    print(f"  Shape B:  {tuple(b.shape)},  dtype: {b.dtype}")
    print(f"  A range:  [{a_f.min().item():.6f}, {a_f.max().item():.6f}]")
    print(f"  B range:  [{b_f.min().item():.6f}, {b_f.max().item():.6f}]")

    print(f"\n--- Absolute Error ---")
    print(f"  Max  Absolute Error:    {max_ae:.6e}")
    print(f"  Mean Absolute Error:    {mean_ae:.6e}")
    print(f"  Median Absolute Error:  {median_ae:.6e}")
    print(f"  RMSE:                   {rmse:.6e}")

    # --- Relative error (avoid div by zero) ---
    denom = a_f.abs().clamp(min=1e-8)
    rel_err = (abs_diff / denom)
    mean_rel = rel_err.mean().item()
    max_rel = rel_err.max().item()
    print(f"\n--- Relative Error ---")
    print(f"  Mean Relative Error:    {mean_rel:.6e}")
    print(f"  Max  Relative Error:    {max_rel:.6e}")

    # --- Cosine similarity (flatten) ---
    a_flat = a_f.reshape(-1)
    b_flat = b_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        a_flat.unsqueeze(0), b_flat.unsqueeze(0)
    ).item()
    print(f"\n--- Cosine Similarity ---")
    print(f"  Cosine Similarity:      {cos_sim:.10f}")

    # --- Pearson correlation ---
    a_mean = a_flat.mean()
    b_mean = b_flat.mean()
    a_centered = a_flat - a_mean
    b_centered = b_flat - b_mean
    pearson = (a_centered * b_centered).sum() / (
        a_centered.norm() * b_centered.norm() + 1e-12
    )
    print(f"  Pearson Correlation:    {pearson.item():.10f}")

    # --- Signal-to-Noise Ratio ---
    signal_power = a_f.pow(2).mean().item()
    noise_power = diff.pow(2).mean().item()
    if noise_power > 0:
        snr_db = 10 * np.log10(signal_power / noise_power)
    else:
        snr_db = float('inf')
    print(f"\n--- SNR ---")
    print(f"  SNR (dB):               {snr_db:.2f}")

    # --- Percentage of elements within tolerance ---
    print(f"\n--- Error Distribution ---")
    total = abs_diff.numel()
    for tol in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]:
        count = (abs_diff <= tol).sum().item()
        pct = count / total * 100
        print(f"  |err| <= {tol:<8g}:  {count:>12d} / {total}  ({pct:6.2f}%)")

    # --- Exact match ---
    exact = (a == b).sum().item()
    print(f"\n  Exact match:            {exact:>12d} / {total}  ({exact/total*100:.2f}%)")

    # --- Per-dimension stats (if >= 2D) ---
    if a.dim() >= 2:
        print(f"\n--- Per-Dimension Error (last dim, first 10 shown) ---")
        # mean absolute error along all dims except last
        dims_to_reduce = list(range(a.dim() - 1))
        per_dim_mae = abs_diff.mean(dim=dims_to_reduce)
        n_show = min(10, per_dim_mae.shape[0])
        for i in range(n_show):
            print(f"  dim[{i}] MAE: {per_dim_mae[i].item():.6e}")
        if per_dim_mae.shape[0] > n_show:
            print(f"  ... ({per_dim_mae.shape[0] - n_show} more dimensions)")

    # --- Per time-step stats (if 3D: T, N, C) ---
    if a.dim() == 3:
        print(f"\n--- Per-Timestep Error (dim=0, first/last 5 shown) ---")
        per_t_mae = abs_diff.mean(dim=(1, 2))
        T = per_t_mae.shape[0]
        n_show = min(5, T)
        for i in range(n_show):
            print(f"  t[{i:>4d}] MAE: {per_t_mae[i].item():.6e}")
        if T > 2 * n_show:
            print(f"  ...")
        for i in range(max(n_show, T - n_show), T):
            print(f"  t[{i:>4d}] MAE: {per_t_mae[i].item():.6e}")

    # --- Per-sample (batch) stats (if 3D: T, N, C, batch=dim1) ---
    if a.dim() == 3:
        N = a.shape[1]
        # MAE per sample: mean over (T, C) dims => shape (N,)
        per_sample_mae = abs_diff.mean(dim=(0, 2))       # (N,)
        per_sample_max = abs_diff.amax(dim=(0, 2))        # (N,)
        per_sample_rmse = diff.pow(2).mean(dim=(0, 2)).sqrt()  # (N,)

        # Cosine similarity per sample
        a_per = a_f.permute(1, 0, 2).reshape(N, -1)  # (N, T*C)
        b_per = b_f.permute(1, 0, 2).reshape(N, -1)  # (N, T*C)
        per_sample_cos = torch.nn.functional.cosine_similarity(a_per, b_per, dim=1)  # (N,)

        # Sort by MAE descending
        sorted_idx = per_sample_mae.argsort(descending=True)

        print(f"\n{'='*70}")
        print(f"Per-Sample (Batch dim=1, N={N}) Analysis")
        print(f"{'='*70}")
        print(f"  {'Rank':<6} {'Sample':<8} {'MAE':>12} {'MaxAE':>12} {'RMSE':>12} {'CosSim':>12}")
        print(f"  {'-'*62}")

        # Top 20 worst
        n_top = min(20, N)
        print(f"\n  >>> Top {n_top} WORST samples (highest MAE):")
        for rank, idx in enumerate(sorted_idx[:n_top]):
            i = idx.item()
            print(f"  {rank+1:<6} {i:<8} {per_sample_mae[i].item():>12.6e} "
                  f"{per_sample_max[i].item():>12.6e} "
                  f"{per_sample_rmse[i].item():>12.6e} "
                  f"{per_sample_cos[i].item():>12.8f}")

        # Top 5 best
        print(f"\n  >>> Top 5 BEST samples (lowest MAE):")
        for rank, idx in enumerate(sorted_idx[-5:].flip(0)):
            i = idx.item()
            print(f"  {rank+1:<6} {i:<8} {per_sample_mae[i].item():>12.6e} "
                  f"{per_sample_max[i].item():>12.6e} "
                  f"{per_sample_rmse[i].item():>12.6e} "
                  f"{per_sample_cos[i].item():>12.8f}")

        # Overall stats
        print(f"\n  --- Summary across all {N} samples ---")
        print(f"  MAE   mean={per_sample_mae.mean().item():.6e}  "
              f"std={per_sample_mae.std().item():.6e}  "
              f"min={per_sample_mae.min().item():.6e}  "
              f"max={per_sample_mae.max().item():.6e}")
        print(f"  MaxAE mean={per_sample_max.mean().item():.6e}  "
              f"std={per_sample_max.std().item():.6e}  "
              f"min={per_sample_max.min().item():.6e}  "
              f"max={per_sample_max.max().item():.6e}")
        print(f"  Cos   mean={per_sample_cos.mean().item():.8f}  "
              f"std={per_sample_cos.std().item():.8f}  "
              f"min={per_sample_cos.min().item():.8f}  "
              f"max={per_sample_cos.max().item():.8f}")

        # Detailed look at the WORST sample
        worst_i = sorted_idx[0].item()
        print(f"\n  {'='*60}")
        print(f"  Detailed look at WORST sample index={worst_i}")
        print(f"  {'='*60}")
        a_worst = a_f[:, worst_i, :]  # (T, C)
        b_worst = b_f[:, worst_i, :]
        diff_worst = (a_worst - b_worst).abs()
        # find the (t, c) location of the max error
        flat_idx = diff_worst.argmax().item()
        t_max = flat_idx // a_worst.shape[1]
        c_max = flat_idx % a_worst.shape[1]
        print(f"  Max error location: t={t_max}, c={c_max}")
        print(f"  A[{t_max},{worst_i},{c_max}] = {a_f[t_max, worst_i, c_max].item():.6f}")
        print(f"  B[{t_max},{worst_i},{c_max}] = {b_f[t_max, worst_i, c_max].item():.6f}")
        print(f"  |diff| = {diff_worst[t_max, c_max].item():.6e}")

        # Per-timestep MAE for the worst sample
        worst_t_mae = diff_worst.mean(dim=1)  # (T,)
        worst_t_sorted = worst_t_mae.argsort(descending=True)
        print(f"\n  Top 10 worst timesteps for sample {worst_i}:")
        for rank in range(min(10, worst_t_mae.shape[0])):
            t_i = worst_t_sorted[rank].item()
            print(f"    t={t_i:<5d}  MAE={worst_t_mae[t_i].item():.6e}")

    return {
        "max_ae": max_ae,
        "mean_ae": mean_ae,
        "rmse": rmse,
        "cosine_sim": cos_sim,
        "pearson": pearson.item(),
        "snr_db": snr_db,
    }


def main():
    print(f"File A: {FILE_A}")
    print(f"File B: {FILE_B}")
    print()

    a = load_tensor(FILE_A)
    b = load_tensor(FILE_B)

    # Handle dict case
    if isinstance(a, dict) and isinstance(b, dict):
        common_keys = set(a.keys()) & set(b.keys())
        print(f"\nCommon keys: {common_keys}")
        for k in sorted(common_keys):
            if isinstance(a[k], torch.Tensor) and isinstance(b[k], torch.Tensor):
                if a[k].shape == b[k].shape:
                    compare_tensors(a[k], b[k], name=k)
                else:
                    print(f"\n[{k}] Shape mismatch: {a[k].shape} vs {b[k].shape}")
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            print(f"\n⚠ Shape mismatch: A={a.shape} vs B={b.shape}")
            # Try to compare the common part
            if a.dim() == b.dim():
                common_shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
                slices = tuple(slice(0, s) for s in common_shape)
                print(f"  Comparing common region: {common_shape}")
                compare_tensors(a[slices], b[slices], name="common_region")
            else:
                print("  Cannot compare: different number of dimensions")
                sys.exit(1)
        else:
            compare_tensors(a, b)
    else:
        print(f"Type A: {type(a)}, Type B: {type(b)}")
        print("Cannot compare non-tensor / mismatched types.")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("Done.")


if __name__ == "__main__":
    main()
