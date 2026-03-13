"""
Compare SPU noquant layer outputs (bf16) with GPU noquant bf16 outputs per LSTM layer.

SPU output processing logic:
  1. Each layer has 30 bin files, each [4, 3538944] bf16 (stored as raw bytes)
  2. Read as uint16, shift left 16 bits → float32 (bf16 → fp32 decoding)
  3. Concat 30 files → [30, 4, 3538944]
  4. Take last 3145728 → [30, 4, 3145728]
  5. Reshape → [30, 4, 32, 3, 128, 256]
  6. Permute(0,2,1,4,3,5) → [30,32,4,128,3,256]
  7. Reshape → [960, 512, 768]

GPU: load original model (bf16, no FakeQuant), run first batch, hook LSTM outputs.
"""

import os
import sys
import random
import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from viterbi_wangqiao import (
    Model, load_quant_model, register_fakequant_hooks,
    TrainingDataSet3, FakeQuant, LSTM,
    accuracy, decode_ref,
)
import toml

# ============================================================
# SPU noquant output loading & reshaping (bf16)
# ============================================================

NOQUANT_GROUPS_PER_FILE = 30  # last 2 of 32 groups are padding
GROUP_SIZE = 3 * 128 * 256    # 98304 elements per group


def load_spu_noquant_layer(layer_dir, layer_idx, num_models=30, reverse=False):
    """Load and reshape one SPU noquant layer output (bf16 → float32).

    Each file has 32 groups per channel, but only the first 30 contain valid
    LSTM output (groups 30-31 are padding/garbage).

    Returns: np.ndarray of shape [T, N, C] float32, or None if not enough files.
    """
    files = []
    for m in range(num_models):
        path = os.path.join(layer_dir, f"layer_{layer_idx}_model_{m}.bin")
        if not os.path.exists(path):
            print(f"  [WARN] Missing {path}, layer {layer_idx} has only {m} files")
            return None
        raw_u16 = np.fromfile(path, dtype=np.uint16)
        t_bf16 = torch.from_numpy(raw_u16.view(np.int16)).view(torch.bfloat16)
        fp32 = t_bf16.float().numpy()
        files.append(fp32.reshape(4, -1))  # [4, 3538944]

    if reverse:
        files = files[::-1]
        print(f"    Reverse layer: model order flipped (29→0)")

    stacked = np.stack(files, axis=0)  # [30, 4, 3538944]
    total_per_ch = stacked.shape[2]
    alloc = 32 * GROUP_SIZE  # 3145728 — full allocated region
    assert total_per_ch >= alloc, f"Expected >= {alloc} per channel, got {total_per_ch}"

    # Take the last 32 groups, then keep only the first 30 valid groups
    keep = NOQUANT_GROUPS_PER_FILE * GROUP_SIZE  # 2949120
    region_start = total_per_ch - alloc
    trimmed = stacked[:, :, region_start : region_start + keep]  # [30, 4, 2949120]

    ng = NOQUANT_GROUPS_PER_FILE  # 30
    if layer_idx == 8:
        reshaped = trimmed.reshape(num_models, 4, 128, ng, 3, 256)
        permuted = reshaped.transpose(0, 3, 1, 2, 4, 5)
    else:
        reshaped = trimmed.reshape(num_models, 4, ng, 3, 128, 256)
        permuted = reshaped.transpose(0, 2, 1, 4, 3, 5)  # [30, 30, 4, 128, 3, 256]

    T = num_models * ng   # 900
    N = 4 * 128           # 512
    C = 3 * 256           # 768
    final = permuted.reshape(T, N, C)
    return final


def load_all_spu_noquant_layers(layer_dir, num_layers=9, reverse_layers=None):
    """Load all available SPU noquant layers."""
    if reverse_layers is None:
        reverse_layers = set()
    results = {}
    for i in range(num_layers):
        rev = i in reverse_layers
        print(f"  Loading SPU noquant layer {i} {'(reverse)' if rev else '(forward)'} ...")
        data = load_spu_noquant_layer(layer_dir, i, reverse=rev)
        if data is not None:
            results[i] = data
            nan_cnt = np.isnan(data).sum()
            inf_cnt = np.isinf(data).sum()
            d64 = data.astype(np.float64)
            print(f"    shape={data.shape}, "
                  f"range=[{np.nanmin(d64):.4f}, {np.nanmax(d64):.4f}], "
                  f"std={np.nanstd(d64):.4f}, nan={nan_cnt}, inf={inf_cnt}")
        else:
            print(f"    Skipped (incomplete)")
    return results


# ============================================================
# GPU noquant model: run first batch, capture LSTM bf16 outputs
# ============================================================

def register_lstm_hooks(model):
    """Hook every LSTM inside backbone Sequential blocks to capture bf16 outputs.

    With io_quant model, each LSTM block is nn.Sequential(LSTM, FakeQuant).
    We hook the LSTM sub-module to get the output BEFORE FakeQuant.
    """
    hook_outputs = {}
    hooks = []

    for bb_name, bb_module in model.backbone._modules.items():
        if not isinstance(bb_module, torch.nn.Sequential):
            continue
        for sub_name, sub_module in bb_module._modules.items():
            if isinstance(sub_module, LSTM):
                tag = f"backbone.{bb_name}.{sub_name}"

                def make_hook(name):
                    def hook_fn(module, inp, output):
                        hook_outputs[name] = output.detach().cpu().float().numpy()
                    return hook_fn

                hooks.append(sub_module.register_forward_hook(make_hook(tag)))

    return hook_outputs, hooks


def get_gpu_noquant_lstm_outputs(config, io_quant_path, act_scales_path,
                                 data_dir, device="cuda:0", batch_size=512,
                                 seed=25):
    """Load io_quant model (bf16), hook LSTM outputs (before FakeQuant)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("[GPU-noquant] Loading io_quant model (bf16) ...")
    model = load_quant_model(
        config=config,
        io_quant_path=io_quant_path,
        act_scales_path=act_scales_path,
        device=device,
        bf16=True,
    )

    hook_outputs, hooks = register_lstm_hooks(model)
    print(f"[GPU-noquant] Registered {len(hooks)} LSTM hooks (before FakeQuant)")

    print("[GPU-noquant] Loading validation data ...")
    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(data_dir, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f"[GPU-noquant] Dataset: {len(dataset)} samples, {len(loader)} batches")

    print("[GPU-noquant] Running forward pass (first batch) ...")
    first_batch_targets = []
    with torch.no_grad():
        for data, target, *_ in loader:
            first_batch_targets = list(torch.unbind(target, 0))
            data = data.to(torch.bfloat16).to(device)
            print(f"[GPU-noquant] Input shape: {data.shape}, dtype: {data.dtype}")
            x = model.backbone(data)
            print(f"[GPU-noquant] Backbone output shape: {x.shape}")
            break

    for h in hooks:
        h.remove()

    sorted_keys = sorted(hook_outputs.keys(),
                         key=lambda k: (int(k.split('.')[1]), int(k.split('.')[2])))
    layer_outputs = {}
    for i, key in enumerate(sorted_keys):
        arr = hook_outputs[key]
        layer_outputs[i] = arr
        print(f"[GPU-noquant] LSTM output {i} ({key}): shape={arr.shape}, "
              f"range=[{arr.min():.4f}, {arr.max():.4f}], std={arr.std():.4f}")

    return model, layer_outputs, first_batch_targets


# ============================================================
# Comparison metrics (float)
# ============================================================

def compare_layers(spu_data, gpu_data, layer_idx, label=""):
    """Compare SPU bf16 vs GPU bf16 for a single layer.

    Handles NaN/Inf by masking them out for metric computation.
    """
    spu = spu_data.astype(np.float64)
    gpu = gpu_data.astype(np.float64)

    T_spu, N_spu, C_spu = spu.shape
    T_gpu, N_gpu, C_gpu = gpu.shape

    T_min = min(T_spu, T_gpu)
    if T_spu != T_gpu:
        print(f"  [NOTE] T mismatch: SPU={T_spu}, GPU={T_gpu}. Comparing last {T_min} timesteps.")
        spu_cmp = spu[T_spu - T_min:]
        gpu_cmp = gpu[T_gpu - T_min:]
    else:
        spu_cmp = spu
        gpu_cmp = gpu

    assert spu_cmp.shape == gpu_cmp.shape, \
        f"Shape mismatch after alignment: {spu_cmp.shape} vs {gpu_cmp.shape}"

    valid_mask = np.isfinite(spu_cmp) & np.isfinite(gpu_cmp)
    n_total = spu_cmp.size
    n_valid = valid_mask.sum()
    n_bad = n_total - n_valid
    if n_bad > 0:
        print(f"  [NOTE] {n_bad} elements ({n_bad/n_total*100:.4f}%) are NaN/Inf, "
              f"excluded from metrics")

    spu_v = spu_cmp[valid_mask]
    gpu_v = gpu_cmp[valid_mask]

    diff = spu_v - gpu_v
    abs_diff = np.abs(diff)

    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff) if len(abs_diff) > 0 else np.nan
    rmse = np.sqrt(np.mean(diff ** 2))

    gpu_abs = np.abs(gpu_v)
    rel_err = np.mean(abs_diff / (gpu_abs + 1e-12)) * 100

    cos_sim = np.nan
    norm_prod = np.linalg.norm(spu_v) * np.linalg.norm(gpu_v)
    if norm_prod > 0:
        cos_sim = np.dot(spu_v, gpu_v) / norm_prod

    print(f"  {'Metric':<25} {'Value':>12}")
    print(f"  {'-'*40}")
    print(f"  {'Valid elements':<25} {n_valid:>12d} / {n_total}")
    print(f"  {'MAE':<25} {mae:>12.6f}")
    print(f"  {'RMSE':<25} {rmse:>12.6f}")
    print(f"  {'Max abs error':<25} {max_err:>12.6f}")
    print(f"  {'Mean relative error':<25} {rel_err:>11.4f}%")
    print(f"  {'Cosine similarity':<25} {cos_sim:>12.8f}")
    print(f"  {'SPU mean/std':<25} {np.mean(spu_v):>8.4f} / {np.std(spu_v):.4f}")
    print(f"  {'GPU mean/std':<25} {np.mean(gpu_v):>8.4f} / {np.std(gpu_v):.4f}")

    print(f"  --- Sample-level cosine sim (first 5 batch elements, t=0) ---")
    for n in range(min(5, spu_cmp.shape[1])):
        s = spu_cmp[0, n, :].flatten()
        g = gpu_cmp[0, n, :].flatten()
        m = np.isfinite(s) & np.isfinite(g)
        s, g = s[m], g[m]
        np_s = np.linalg.norm(s) * np.linalg.norm(g)
        cs = np.dot(s, g) / np_s if np_s > 0 else 0
        print(f"    batch[{n}] cos_sim={cs:.8f}  "
              f"SPU[:5]=[{', '.join(f'{v:.4f}' for v in s[:5])}]  "
              f"GPU[:5]=[{', '.join(f'{v:.4f}' for v in g[:5])}]")

    return {
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "rel_err": rel_err,
        "cos_sim": cos_sim,
    }


# ============================================================
# End-to-end accuracy evaluation: bf16 backbone output → decode → accuracy
# ============================================================

def evaluate_from_float(float_data_np, model, targets, device, label=""):
    """bf16 backbone output → crfencoder → decode → accuracy.

    Args:
        float_data_np: np.ndarray [T, N, C] float32 (last layer backbone output)
        model: Model with crfencoder and decode_batch
        targets: list of target tensors (from dataloader)
        device: torch device
        label: label string for printing

    Returns:
        dict with mean, median accuracy
    """
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=0.95)

    x = torch.from_numpy(float_data_np).to(dtype=torch.bfloat16, device=device)

    with torch.no_grad():
        scores = model.crfencoder(x)
        decoded = model.decode_batch(scores)

    refs = [decode_ref(t, model.alphabet) for t in targets]
    accs = [accuracy_with_cov(r, s) if len(s) else 0.0 for r, s in zip(refs, decoded)]

    mean_acc = np.mean(accs)
    median_acc = np.median(accs)
    print(f"  [{label}] mean={mean_acc:.2f}%  median={median_acc:.2f}%  "
          f"chunks={len(refs)}")
    return {"mean": mean_acc, "median": median_acc}


# ============================================================
# Main
# ============================================================

def main():
    CONFIG_PATH = "/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/config.toml"
    IO_QUANT_PATH = os.path.join(SCRIPT_DIR, "..", "caoyu", "layer_9_6x_io_quant_0305.pth")
    ACT_SCALES_PATH = os.path.join(SCRIPT_DIR, "..", "caoyu", "layer_9_6x_act_scales_0305.pth")
    DATA_DIR = "/workspace/huada/moffett_data/250F600274011_train_data/val"
    DEVICE = "cuda:0"

    NOQUANT_DIR = os.path.join(SCRIPT_DIR, "noquant_layer_output")

    config = toml.load(CONFIG_PATH)
    num_layers = config["encoder"]["num_layers"]
    print(f"Model has {num_layers} LSTM layers")

    reverse_layers = {i for i in range(num_layers) if (num_layers - i) % 2}
    print(f"Reverse layers (model order flipped): {sorted(reverse_layers)}")
    print()

    # ---- Step 1: Load SPU noquant outputs ----
    print("=" * 70)
    print("Loading SPU noquant outputs (bf16)")
    print("=" * 70)
    spu_noquant = load_all_spu_noquant_layers(NOQUANT_DIR, num_layers, reverse_layers)

    # ---- Step 2: Get GPU noquant bf16 outputs ----
    print()
    print("=" * 70)
    print("Running GPU noquant model (bf16, no FakeQuant)")
    print("=" * 70)
    model, gpu_noquant, first_batch_targets = get_gpu_noquant_lstm_outputs(
        config=config,
        io_quant_path=IO_QUANT_PATH,
        act_scales_path=ACT_SCALES_PATH,
        data_dir=DATA_DIR,
        device=DEVICE,
        batch_size=512,
    )

    # ---- Step 3: Compare per layer ----
    print()
    print("=" * 70)
    print("Per-layer comparison: SPU noquant vs GPU noquant (bf16)")
    print("=" * 70)
    noquant_results = {}
    for layer_idx in sorted(set(list(spu_noquant.keys()) + list(gpu_noquant.keys()))):
        if layer_idx not in spu_noquant:
            print(f"\n--- Layer {layer_idx}: SPU noquant data missing, skipped ---")
            continue
        if layer_idx not in gpu_noquant:
            print(f"\n--- Layer {layer_idx}: GPU noquant data missing, skipped ---")
            continue
        print(f"\n--- Layer {layer_idx} (SPU noquant vs GPU noquant) ---")
        print(f"  SPU shape: {spu_noquant[layer_idx].shape}, "
              f"GPU shape: {gpu_noquant[layer_idx].shape}")
        noquant_results[layer_idx] = compare_layers(
            spu_noquant[layer_idx], gpu_noquant[layer_idx], layer_idx, "noquant")

    # ---- Step 4: End-to-end accuracy (decode last layer → accuracy) ----
    last_layer_idx = num_layers - 1
    print()
    print("=" * 70)
    print(f"End-to-end accuracy evaluation (last layer {last_layer_idx} → crfencoder → decode)")
    print("=" * 70)

    eval_results = {}
    if last_layer_idx in gpu_noquant:
        print(f"\n  GPU noquant (bf16 → crfencoder → decode):")
        eval_results["GPU-noquant"] = evaluate_from_float(
            gpu_noquant[last_layer_idx], model,
            first_batch_targets, DEVICE, label="GPU-noquant")

    if last_layer_idx in spu_noquant:
        print(f"\n  SPU noquant (bf16 → crfencoder → decode):")
        eval_results["SPU-noquant"] = evaluate_from_float(
            spu_noquant[last_layer_idx], model,
            first_batch_targets, DEVICE, label="SPU-noquant")

    # ---- Summary table ----
    print()
    print("=" * 70)
    print("SUMMARY: Noquant (bf16) — SPU vs GPU per-layer")
    print("=" * 70)
    header = (f"{'Layer':>5} | {'MAE':>10} | {'RMSE':>10} | {'MaxErr':>10} | "
              f"{'RelErr%':>10} | {'CosSim':>12}")
    print(header)
    print("-" * len(header))
    for layer_idx in range(num_layers):
        if layer_idx in noquant_results:
            r = noquant_results[layer_idx]
            print(f"{layer_idx:>5} | {r['mae']:>10.6f} | {r['rmse']:>10.6f} | "
                  f"{r['max_err']:>10.6f} | {r['rel_err']:>9.4f}% | "
                  f"{r['cos_sim']:>12.8f}")

    if eval_results:
        print()
        print("=" * 70)
        print("ACCURACY (last layer → crfencoder → decode, first batch)")
        print("=" * 70)
        for name, r in eval_results.items():
            print(f"  {name:>14}:  mean={r['mean']:.2f}%  median={r['median']:.2f}%")


if __name__ == "__main__":
    main()
