"""
Load quantized model and run inference on placeholder_backend.npz input.

Usage:
    python /workspace/huada/scall/caoyu/run_backend.py [--device cuda:0]
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add project root so we can import from viterbi_0224
pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, pro_dir)

import toml
from viterbi_0224 import (
    Model, load_quant_model, insert_fakequant_backbone, FakeQuant,
    match_names,
)

# ============================================================
# Paths
# ============================================================
CAOYU_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/config.toml"
IO_QUANT_PATH = os.path.join(CAOYU_DIR, "layer_9_6x_io_quant_0303.pth")
ACT_SCALES_PATH = os.path.join(CAOYU_DIR, "layer_9_6x_act_scales_0303.pth")
NPZ_PATH = os.path.join(CAOYU_DIR, "placeholder_backend.npz")


def bf16_to_float32(tensor: np.ndarray):
    """Convert bf16 stored as uint16 to float32 by left-shifting 16 bits."""
    tsr = np.left_shift(tensor.view("uint16").astype(np.int32), 16).view(np.float32)
    return tsr


def load_npz_input(npz_path):
    """
    Load input from placeholder_backend.npz.
    The 'input' field is stored as uint16 (bfloat16 bit pattern), shape (128, 1, 4800, 8).
    Only the first element of the last dim contains data; convert to float32.
    """
    data = np.load(npz_path, allow_pickle=True)
    inp_raw = data["input"]  # (128, 1, 4800, 8) uint16
    # Take only the first column (rest are zero-padding)
    inp_u16 = inp_raw[:, :, :4796, :1].copy()  # (128, 1, 4796, 1) uint16
    # Convert bf16 (uint16) -> float32
    inp_fp32 = bf16_to_float32(inp_u16)  # (128, 1, 4796, 1) float32
    inp_tensor = torch.from_numpy(inp_fp32)
    print(f"[Input] shape={inp_tensor.shape}, dtype={inp_tensor.dtype}, "
          f"min={inp_tensor.min().item():.4f}, max={inp_tensor.max().item():.4f}, "
          f"mean={inp_tensor.mean().item():.4f}, std={inp_tensor.std().item():.4f}")
    return inp_tensor


def register_hooks(model):
    """Register forward hooks on every leaf module to capture outputs."""
    hook_outputs = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            # For LSTM, output is (y, (h, c)) — take y only
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            hook_outputs[name] = out.detach().cpu()
        return hook_fn

    # Hook backbone sub-layers (including Sequential internals)
    for name, module in model.backbone.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            full_name = f"backbone.{name}" if name else "backbone"
            h = module.register_forward_hook(make_hook(full_name))
            hooks.append(h)

    # Hook crfencoder sub-layers
    for name, module in model.crfencoder.named_modules():
        if len(list(module.children())) == 0:
            full_name = f"crfencoder.{name}" if name else "crfencoder"
            h = module.register_forward_hook(make_hook(full_name))
            hooks.append(h)

    # Also hook the top-level backbone and crfencoder themselves
    hooks.append(model.backbone.register_forward_hook(make_hook("backbone")))
    hooks.append(model.crfencoder.register_forward_hook(make_hook("crfencoder")))

    return hook_outputs, hooks


def main():
    parser = argparse.ArgumentParser(description="Run quantized model on placeholder_backend input")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = args.device

    # ---- Load config ----
    assert os.path.exists(CONFIG_PATH), f"Config not found: {CONFIG_PATH}"
    config = toml.load(CONFIG_PATH)
    print(f"[Config] loaded from {CONFIG_PATH}")

    # ---- Load quantized model (all LSTM I/O quantized) ----
    print("[Loading quantized model]")
    model = load_quant_model(
        config=config,
        io_quant_path=IO_QUANT_PATH,
        act_scales_path=ACT_SCALES_PATH,
        device=device,
        half=True,
    )
    model.eval()
    print(f"[Model] loaded (fp16)")
    print(model)

    # ---- Register hooks ----
    hook_outputs, hooks = register_hooks(model)
    print(f"\n[Hooks registered] {len(hooks)} hooks")

    # ---- Load input ----
    print("\n[Loading input from npz]")
    inp = load_npz_input(NPZ_PATH)              # (128, 1, 4796, 1)
    inp = inp.squeeze(-1).half().to(device)      # (128, 1, 4796) fp16

    # ---- Forward pass ----
    print("\n[Running forward pass (fp16)]")
    with torch.no_grad():
        scores = model(inp)

    # ---- Remove hooks ----
    for h in hooks:
        h.remove()

    # ---- Print captured outputs ----
    print(f"\n[Captured {len(hook_outputs)} layer outputs]")
    print(f"{'Layer':<45} {'Shape':<30} {'dtype':<12} {'min':>10} {'max':>10}")
    print("-" * 110)
    for name, tensor in hook_outputs.items():
        print(f"{name:<45} {str(tuple(tensor.shape)):<30} {str(tensor.dtype):<12} "
              f"{tensor.float().min().item():>10.4f} {tensor.float().max().item():>10.4f}")

    # ---- Save all layer outputs ----
    output_path = os.path.join(CAOYU_DIR, "layer_outputs.npz")
    save_dict = {}
    for name, tensor in hook_outputs.items():
        # Use safe key: replace '.' with '__'
        key = name.replace(".", "__")
        save_dict[key] = tensor.numpy()
    np.savez(output_path, **save_dict)
    print(f"\n[Saved] {output_path}  ({len(save_dict)} layers)")


if __name__ == "__main__":
    main()
