"""
Compare SPU quant layer outputs (int8) with GPU FakeQuant int8 outputs per LSTM layer.

SPU output processing logic:
  1. Each layer has 30 bin files, each [4, 3538944] int8
  2. Concat 30 files → [30, 4, 3538944]
  3. Take last 3145728 → [30, 4, 3145728]
  4. Reshape → [30, 4, 32, 3, 128, 256]
  5. Permute(0,2,1,4,3,5) → [30,32,4,128,3,256]
  6. Reshape → [960, 512, 768]

GPU: load FakeQuant model (bf16), run first batch, hook int8 outputs.
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
# SPU output loading & reshaping
# ============================================================

SPU_BACKBONE_INT8_PT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "quant_layer_output", "spu_backbone_output_quant_int8.pt",
)


def load_spu_layer(layer_dir, layer_idx, num_models=30, reverse=False):
    """Load and reshape one SPU layer's output.

    Returns: np.ndarray of shape [T, N, C] int8, or None if not enough files.
    Layer 8 is loaded directly from a pre-saved .pt file.
    """
    if layer_idx == 8:
        final = torch.load(SPU_BACKBONE_INT8_PT)
        if isinstance(final, torch.Tensor):
            final = final.numpy()
        return final

    files = []
    for m in range(num_models):
        path = os.path.join(layer_dir, f"layer_{layer_idx}_model_{m}.bin")
        if not os.path.exists(path):
            print(f"  [WARN] Missing {path}, layer {layer_idx} has only {m} files")
            return None
        raw = np.fromfile(path, dtype=np.int8)
        files.append(raw.reshape(4, -1))  # [4, 3538944]

    if reverse:
        files = files[::-1]
        print(f"    Reverse layer: model order flipped (29→0)")

    stacked = np.stack(files, axis=0)  # [30, 4, 3538944]
    total_per_ch = stacked.shape[2]
    keep = 3145728
    assert total_per_ch >= keep, f"Expected >= {keep} per channel, got {total_per_ch}"

    trimmed = stacked[:, :, total_per_ch - keep:]  # [30, 4, 3145728]

    reshaped = trimmed.reshape(30, 4, 32, 3, 128, 256)
    permuted = reshaped.transpose(0, 2, 1, 4, 3, 5)  # [30, 32, 4, 128, 3, 256]
    final = permuted.reshape(960, 512, 768)  # [T=960, N=512, C=768]
    return final


def load_all_spu_layers(layer_dir, num_layers=9, reverse_layers=None):
    """Load all available SPU layers."""
    if reverse_layers is None:
        reverse_layers = set()
    results = {}
    for i in range(num_layers):
        rev = i in reverse_layers
        print(f"  Loading SPU layer {i} {'(reverse)' if rev else '(forward)'} ...")
        data = load_spu_layer(layer_dir, i, reverse=rev)
        if data is not None:
            results[i] = data
            print(f"    shape={data.shape}, range=[{data.min()}, {data.max()}]")
        else:
            print(f"    Skipped (incomplete)")
    return results


# ============================================================
# GPU model: run first batch, capture FakeQuant int8 outputs
# ============================================================

def get_gpu_fakequant_int8(config, io_quant_path, act_scales_path,
                           data_dir, device="cuda:0", batch_size=512, seed=25):
    """Run GPU FakeQuant model on first batch, return per-layer int8 outputs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("[GPU] Loading quantized model (bf16) ...")
    model = load_quant_model(
        config=config,
        io_quant_path=io_quant_path,
        act_scales_path=act_scales_path,
        device=device,
        bf16=True,
    )

    hook_outputs, hooks = register_fakequant_hooks(model)
    print(f"[GPU] Registered {len(hooks)} FakeQuant hooks")

    print("[GPU] Loading validation data ...")
    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(data_dir, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    print(f"[GPU] Dataset: {len(dataset)} samples, {len(loader)} batches")

    print("[GPU] Running forward pass (first batch) ...")
    with torch.no_grad():
        for data, target, *_ in loader:
            data = data.to(torch.bfloat16).to(device)
            print(f"[GPU] Input shape: {data.shape}, dtype: {data.dtype}")
            x = model.backbone(data)
            print(f"[GPU] Backbone output shape: {x.shape}")
            break

    for h in hooks:
        h.remove()

    # Sort hooks by backbone module index, then sub-module index
    sorted_keys = sorted(hook_outputs.keys(),
                         key=lambda k: (int(k.split('.')[1]), int(k.split('.')[2])))
    print(f"[GPU] All hooks ({len(sorted_keys)}):")
    for key in sorted_keys:
        tensor = hook_outputs[key]
        print(f"  {key}: shape={tensor.shape}, "
              f"range=[{tensor.float().min().item()}, {tensor.float().max().item()}], "
              f"std={tensor.float().std().item():.2f}")

    # If the first LSTM block has 2 FakeQuants (input + output), the first
    # hook is the input FakeQuant and should be skipped.  If it has only 1,
    # all hooks are output FakeQuants.
    first_bb_idx = sorted_keys[0].split('.')[1]
    has_input_fq = sum(1 for k in sorted_keys if k.split('.')[1] == first_bb_idx) >= 2
    lstm_output_keys = sorted_keys[1:] if has_input_fq else sorted_keys
    if has_input_fq:
        print(f"[GPU] Detected input FakeQuant at {sorted_keys[0]}, skipping it")

    layer_outputs = {}
    for i, key in enumerate(lstm_output_keys):
        tensor = hook_outputs[key]
        layer_outputs[i] = tensor.numpy()
        print(f"[GPU] LSTM output {i} ({key}): shape={tensor.shape}")

    return layer_outputs


def get_gpu_fakequant_int8_with_spu_inject(config, io_quant_path, act_scales_path,
                                            data_dir, spu_int8_data, inject_layer,
                                            device="cuda:0", batch_size=512, seed=25):
    """Run GPU FakeQuant model but inject SPU output at a given layer.

    Replaces the output of LSTM layer `inject_layer` with SPU's int8 data
    (dequantized via that layer's output FakeQuant scale), then continues the
    forward pass through remaining layers on GPU.

    Returns: dict of per-layer int8 outputs (from hooks).
    """
    tag = f"GPU+SPU_L{inject_layer}"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[{tag}] Loading quantized model (bf16) ...")
    model = load_quant_model(
        config=config,
        io_quant_path=io_quant_path,
        act_scales_path=act_scales_path,
        device=device,
        bf16=True,
    )

    hook_outputs, hooks = register_fakequant_hooks(model)
    print(f"[{tag}] Registered {len(hooks)} FakeQuant hooks")

    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(data_dir, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    bb_keys = list(model.backbone._modules.keys())
    lstm_keys = [k for k in bb_keys
                 if isinstance(model.backbone._modules[k], torch.nn.Sequential)]
    assert inject_layer < len(lstm_keys), \
        f"inject_layer={inject_layer} but only {len(lstm_keys)} LSTM blocks"
    inject_key = lstm_keys[inject_layer]
    print(f"[{tag}] Inject at backbone.{inject_key} (LSTM layer {inject_layer})")

    inject_block = model.backbone._modules[inject_key]
    output_fq = None
    for sub in inject_block:
        if isinstance(sub, FakeQuant):
            output_fq = sub
    assert output_fq is not None
    inject_scale = output_fq.scale
    print(f"[{tag}] Layer {inject_layer} output FakeQuant scale shape: {inject_scale.shape}")

    with torch.no_grad():
        for data, target, *_ in loader:
            data = data.to(torch.bfloat16).to(device)
            print(f"[{tag}] Input shape: {data.shape}")

            x = data
            for key in bb_keys:
                mod = model.backbone._modules[key]
                x = mod(x)
                if key == inject_key:
                    T_gpu = x.shape[0]
                    spu_t = torch.from_numpy(spu_int8_data.astype(np.float32))
                    T_spu = spu_t.shape[0]
                    T_min = min(T_spu, T_gpu)
                    spu_inject = spu_t[T_spu - T_min:].to(dtype=x.dtype, device=x.device)
                    scale_val = inject_scale.to(dtype=x.dtype, device=x.device)
                    x_new = spu_inject * scale_val
                    if T_gpu > T_min:
                        x[T_gpu - T_min:] = x_new
                    else:
                        x = x_new
                    print(f"[{tag}] Injected SPU layer {inject_layer} "
                          f"(T_spu={T_spu}, T_gpu={T_gpu}, used last {T_min})")
            print(f"[{tag}] Backbone output shape: {x.shape}")
            break

    for h in hooks:
        h.remove()

    sorted_keys = sorted(hook_outputs.keys(),
                         key=lambda k: (int(k.split('.')[1]), int(k.split('.')[2])))
    first_bb_idx = sorted_keys[0].split('.')[1]
    has_input_fq = sum(1 for k in sorted_keys if k.split('.')[1] == first_bb_idx) >= 2
    lstm_output_keys = sorted_keys[1:] if has_input_fq else sorted_keys

    layer_outputs = {}
    for i, key in enumerate(lstm_output_keys):
        tensor = hook_outputs[key]
        layer_outputs[i] = tensor.numpy()
        print(f"[{tag}] LSTM output {i} ({key}): shape={tensor.shape}")

    return layer_outputs


# ============================================================
# Comparison metrics
# ============================================================

def compare_layers(spu_data, gpu_data, layer_idx, label=""):
    """Compare SPU int8 vs GPU int8 for a single layer."""
    spu = spu_data.astype(np.float32)
    gpu = gpu_data.astype(np.float32)

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

    diff = spu_cmp - gpu_cmp
    abs_diff = np.abs(diff)

    exact_match = np.mean(diff == 0) * 100
    within_1 = np.mean(abs_diff <= 1) * 100
    within_2 = np.mean(abs_diff <= 2) * 100
    within_5 = np.mean(abs_diff <= 5) * 100
    mae = np.mean(abs_diff)
    max_err = np.max(abs_diff)
    rmse = np.sqrt(np.mean(diff ** 2))

    cos_sim = np.nan
    spu_flat = spu_cmp.flatten()
    gpu_flat = gpu_cmp.flatten()
    norm_prod = np.linalg.norm(spu_flat) * np.linalg.norm(gpu_flat)
    if norm_prod > 0:
        cos_sim = np.dot(spu_flat, gpu_flat) / norm_prod

    print(f"  {'Metric':<25} {'Value':>12}")
    print(f"  {'-'*40}")
    print(f"  {'Exact match':<25} {exact_match:>11.2f}%")
    print(f"  {'Within ±1':<25} {within_1:>11.2f}%")
    print(f"  {'Within ±2':<25} {within_2:>11.2f}%")
    print(f"  {'Within ±5':<25} {within_5:>11.2f}%")
    print(f"  {'MAE':<25} {mae:>12.4f}")
    print(f"  {'RMSE':<25} {rmse:>12.4f}")
    print(f"  {'Max abs error':<25} {max_err:>12.0f}")
    print(f"  {'Cosine similarity':<25} {cos_sim:>12.6f}")
    print(f"  {'SPU mean/std':<25} {np.mean(spu_cmp):>7.2f} / {np.std(spu_cmp):.2f}")
    print(f"  {'GPU mean/std':<25} {np.mean(gpu_cmp):>7.2f} / {np.std(gpu_cmp):.2f}")

    # Per-sample cosine similarity (first 5 samples)
    print(f"  --- Sample-level cosine sim (first 5 batch elements, t=0) ---")
    for n in range(min(5, spu_cmp.shape[1])):
        s = spu_cmp[0, n, :].flatten()
        g = gpu_cmp[0, n, :].flatten()
        np_s = np.linalg.norm(s) * np.linalg.norm(g)
        cs = np.dot(s, g) / np_s if np_s > 0 else 0
        print(f"    batch[{n}] cos_sim={cs:.6f}  SPU[:5]={s[:5]}  GPU[:5]={g[:5]}")

    return {
        "exact_match": exact_match,
        "within_1": within_1,
        "within_2": within_2,
        "within_5": within_5,
        "mae": mae,
        "rmse": rmse,
        "max_err": max_err,
        "cos_sim": cos_sim,
    }


# ============================================================
# End-to-end accuracy evaluation: int8 backbone output → decode → accuracy
# ============================================================

def get_last_layer_scale(model):
    """Get the FakeQuant output scale of the last LSTM layer in backbone."""
    bb_keys = list(model.backbone._modules.keys())
    last_lstm_key = None
    for k in reversed(bb_keys):
        m = model.backbone._modules[k]
        if isinstance(m, torch.nn.Sequential):
            last_lstm_key = k
            break
    assert last_lstm_key is not None
    last_block = model.backbone._modules[last_lstm_key]
    output_fq = None
    for sub in last_block:
        if isinstance(sub, FakeQuant):
            output_fq = sub
    assert output_fq is not None
    return output_fq.scale


def evaluate_from_int8(int8_data_np, model, scale, targets, device, label=""):
    """Dequantize int8 backbone output → crfencoder → decode → accuracy.

    Args:
        int8_data_np: np.ndarray [T, N, C] int8 (last layer backbone output)
        model: Model with crfencoder and decode_batch
        scale: FakeQuant scale tensor for the last layer
        targets: list of target tensors (from dataloader)
        device: torch device
        label: label string for printing

    Returns:
        dict with mean, median accuracy
    """
    accuracy_with_cov = lambda ref, seq: accuracy(ref, seq, min_coverage=0.95)

    x = torch.from_numpy(int8_data_np.astype(np.float32)).to(dtype=torch.bfloat16, device=device)
    scale_val = scale.to(dtype=torch.bfloat16, device=device)
    x = x * scale_val

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

    QUANT_DIR = os.path.join(SCRIPT_DIR, "quant_layer_output")

    config = toml.load(CONFIG_PATH)
    num_layers = config["encoder"]["num_layers"]
    print(f"Model has {num_layers} LSTM layers")

    reverse_layers = {i for i in range(num_layers) if (num_layers - i) % 2}
    print(f"Reverse layers (model order flipped): {sorted(reverse_layers)}")
    print()

    # ---- Step 1: Load SPU quant outputs ----
    print("=" * 70)
    print("Loading SPU quant outputs")
    print("=" * 70)
    spu_quant = load_all_spu_layers(QUANT_DIR, num_layers, reverse_layers)

    # ---- Step 2: Get GPU FakeQuant int8 outputs + keep model/targets for eval ----
    print()
    print("=" * 70)
    print("Running GPU FakeQuant model")
    print("=" * 70)
    torch.manual_seed(25)
    torch.cuda.manual_seed_all(25)
    np.random.seed(25)
    random.seed(25)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = load_quant_model(
        config=config,
        io_quant_path=IO_QUANT_PATH,
        act_scales_path=ACT_SCALES_PATH,
        device=DEVICE,
        bf16=True,
    )
    last_layer_scale = get_last_layer_scale(model)
    print(f"Last layer FakeQuant scale: {last_layer_scale.shape}")

    from torch.utils.data import DataLoader
    dataset = TrainingDataSet3(DATA_DIR, tokenization="kmer")
    loader = DataLoader(dataset, batch_size=512, shuffle=False,
                        num_workers=0, pin_memory=True)

    hook_outputs, hooks = register_fakequant_hooks(model)
    first_batch_targets = []
    with torch.no_grad():
        for data, target, *_ in loader:
            first_batch_targets = list(torch.unbind(target, 0))
            data = data.to(torch.bfloat16).to(DEVICE)
            x = model.backbone(data)
            print(f"[GPU] Backbone output shape: {x.shape}")
            break
    for h in hooks:
        h.remove()

    sorted_keys = sorted(hook_outputs.keys(),
                         key=lambda k: (int(k.split('.')[1]), int(k.split('.')[2])))
    first_bb_idx = sorted_keys[0].split('.')[1]
    has_input_fq = sum(1 for k in sorted_keys if k.split('.')[1] == first_bb_idx) >= 2
    lstm_output_keys = sorted_keys[1:] if has_input_fq else sorted_keys
    if has_input_fq:
        print(f"[GPU] Detected input FakeQuant at {sorted_keys[0]}, skipping it")
    else:
        print(f"[GPU] No input FakeQuant detected, all hooks are output FakeQuants")

    gpu_int8 = {}
    for i, key in enumerate(lstm_output_keys):
        tensor = hook_outputs[key]
        gpu_int8[i] = tensor.numpy()
        print(f"[GPU] LSTM output {i} ({key}): shape={tensor.shape}")

    # ---- Step 3: Compare per layer ----
    print()
    print("=" * 70)
    print("Per-layer comparison: SPU quant vs GPU FakeQuant int8")
    print("=" * 70)
    quant_results = {}
    for layer_idx in sorted(set(list(spu_quant.keys()) + list(gpu_int8.keys()))):
        if layer_idx not in spu_quant:
            print(f"\n--- Layer {layer_idx}: SPU quant data missing, skipped ---")
            continue
        if layer_idx not in gpu_int8:
            print(f"\n--- Layer {layer_idx}: GPU data missing, skipped ---")
            continue
        print(f"\n--- Layer {layer_idx} (SPU quant vs GPU) ---")
        print(f"  SPU shape: {spu_quant[layer_idx].shape}, GPU shape: {gpu_int8[layer_idx].shape}")
        quant_results[layer_idx] = compare_layers(
            spu_quant[layer_idx], gpu_int8[layer_idx], layer_idx, "quant")

    # ---- Step 4: Injection experiments ----
    inject_layers = [0, num_layers - 2]  # L0 and L7 (second-to-last)
    gpu_injected_all = {}  # {inject_layer: {layer_idx: int8_np}}
    inject_results_all = {}  # {inject_layer: {layer_idx: metrics}}

    for inj_layer in inject_layers:
        if inj_layer not in spu_quant:
            print(f"\n[SKIP] SPU layer {inj_layer} not available, skipping injection")
            gpu_injected_all[inj_layer] = {}
            inject_results_all[inj_layer] = {}
            continue

        print()
        print("=" * 70)
        print(f"Experiment: Inject SPU layer {inj_layer} → GPU, "
              f"then GPU runs layers {inj_layer+1}+")
        print("=" * 70)
        gpu_inj = get_gpu_fakequant_int8_with_spu_inject(
            config=config,
            io_quant_path=IO_QUANT_PATH,
            act_scales_path=ACT_SCALES_PATH,
            data_dir=DATA_DIR,
            spu_int8_data=spu_quant[inj_layer],
            inject_layer=inj_layer,
            device=DEVICE,
            batch_size=512,
        )
        gpu_injected_all[inj_layer] = gpu_inj

        label = f"SPU_L{inj_layer}"
        print()
        print("=" * 70)
        print(f"Per-layer comparison: SPU quant vs GPU({label} injected)")
        print("=" * 70)
        inj_results = {}
        for layer_idx in sorted(set(list(spu_quant.keys()) + list(gpu_inj.keys()))):
            if layer_idx <= inj_layer:
                continue
            if layer_idx not in spu_quant:
                print(f"\n--- Layer {layer_idx}: SPU quant data missing, skipped ---")
                continue
            if layer_idx not in gpu_inj:
                print(f"\n--- Layer {layer_idx}: GPU(injected) data missing, skipped ---")
                continue
            print(f"\n--- Layer {layer_idx} (SPU quant vs GPU+{label}) ---")
            print(f"  SPU shape: {spu_quant[layer_idx].shape}, "
                  f"GPU(injected) shape: {gpu_inj[layer_idx].shape}")
            inj_results[layer_idx] = compare_layers(
                spu_quant[layer_idx], gpu_inj[layer_idx], layer_idx, label)
        inject_results_all[inj_layer] = inj_results

    # ---- Step 5: End-to-end accuracy (decode last layer → accuracy) ----
    last_layer_idx = num_layers - 1
    print()
    print("=" * 70)
    print(f"End-to-end accuracy evaluation (last layer {last_layer_idx} → crfencoder → decode)")
    print("=" * 70)

    eval_results = {}
    if last_layer_idx in gpu_int8:
        print(f"\n  GPU FakeQuant (int8 → dequant → crfencoder → decode):")
        eval_results["GPU"] = evaluate_from_int8(
            gpu_int8[last_layer_idx], model, last_layer_scale,
            first_batch_targets, DEVICE, label="GPU")

    for inj_layer in inject_layers:
        gpu_inj = gpu_injected_all.get(inj_layer, {})
        if last_layer_idx in gpu_inj:
            label = f"GPU+SPU_L{inj_layer}"
            print(f"\n  {label} (inject SPU layer{inj_layer} → GPU runs rest → decode):")
            eval_results[label] = evaluate_from_int8(
                gpu_inj[last_layer_idx], model, last_layer_scale,
                first_batch_targets, DEVICE, label=label)

    if last_layer_idx in spu_quant:
        print(f"\n  SPU quant (int8 → dequant → crfencoder → decode):")
        eval_results["SPU"] = evaluate_from_int8(
            spu_quant[last_layer_idx], model, last_layer_scale,
            first_batch_targets, DEVICE, label="SPU")

    # ---- Summary table ----
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    header = (f"{'Layer':>5} | {'Experiment':>20} | {'Exact%':>8} | {'±1%':>8} | "
              f"{'±2%':>8} | {'MAE':>8} | {'RMSE':>8} | {'CosSim':>8}")
    print(header)
    print("-" * len(header))
    for layer_idx in range(num_layers):
        if layer_idx in quant_results:
            r = quant_results[layer_idx]
            print(f"{layer_idx:>5} | {'SPU vs GPU':>20} | {r['exact_match']:>7.2f}% | "
                  f"{r['within_1']:>7.2f}% | {r['within_2']:>7.2f}% | "
                  f"{r['mae']:>8.4f} | {r['rmse']:>8.4f} | {r['cos_sim']:>8.6f}")
        for inj_layer in inject_layers:
            inj_results = inject_results_all.get(inj_layer, {})
            if layer_idx in inj_results:
                tag = f"SPU vs GPU+SPU_L{inj_layer}"
                r = inj_results[layer_idx]
                print(f"{layer_idx:>5} | {tag:>20} | {r['exact_match']:>7.2f}% | "
                      f"{r['within_1']:>7.2f}% | {r['within_2']:>7.2f}% | "
                      f"{r['mae']:>8.4f} | {r['rmse']:>8.4f} | {r['cos_sim']:>8.6f}")

    if eval_results:
        print()
        print("=" * 70)
        print("ACCURACY (last layer → crfencoder → decode, first batch)")
        print("=" * 70)
        for name, r in eval_results.items():
            print(f"  {name:>14}:  mean={r['mean']:.2f}%  median={r['median']:.2f}%")


if __name__ == "__main__":
    main()
