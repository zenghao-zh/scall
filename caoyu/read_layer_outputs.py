"""
Read and inspect layer outputs saved by run_backend.py.

Usage:
    # Show summary of all layers
    python /workspace/huada/scall/caoyu/read_layer_outputs.py

    # Show detailed stats for a specific layer (use __ instead of .)
    python /workspace/huada/scall/caoyu/read_layer_outputs.py --layer backbone__4__1__rnn

    # List all available layer keys
    python /workspace/huada/scall/caoyu/read_layer_outputs.py --list
"""

import os
import argparse
import numpy as np

CAOYU_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NPZ = os.path.join(CAOYU_DIR, "layer_outputs.npz")


def key_to_name(key):
    """Convert npz key (backbone__4__1__rnn) back to layer name (backbone.4.1.rnn)."""
    return key.replace("__", ".")


def print_summary(data):
    """Print a summary table of all layers."""
    print(f"{'Layer':<45} {'Shape':<30} {'dtype':<10} {'min':>10} {'max':>10} {'mean':>10} {'std':>10}")
    print("-" * 130)
    for key in data.files:
        arr = data[key]
        fp = arr.astype(np.float32)
        name = key_to_name(key)
        print(f"{name:<45} {str(arr.shape):<30} {str(arr.dtype):<10} "
              f"{fp.min():>10.4f} {fp.max():>10.4f} {fp.mean():>10.4f} {fp.std():>10.4f}")


def print_detail(data, key):
    """Print detailed statistics for a single layer."""
    arr = data[key]
    fp = arr.astype(np.float32)
    name = key_to_name(key)

    print(f"Layer: {name}")
    print(f"  Key:   {key}")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Min:   {fp.min():.6f}")
    print(f"  Max:   {fp.max():.6f}")
    print(f"  Mean:  {fp.mean():.6f}")
    print(f"  Std:   {fp.std():.6f}")

    # Per-dimension stats for the last axis
    ndim = arr.ndim
    if ndim >= 2:
        # Flatten all dims except last → (*, last_dim)
        flat = fp.reshape(-1, fp.shape[-1])
        ch_mean = flat.mean(axis=0)
        ch_std = flat.std(axis=0)
        print(f"\n  Per-channel (last dim, size={fp.shape[-1]}) stats:")
        print(f"    channel mean:  min={ch_mean.min():.6f}  max={ch_mean.max():.6f}")
        print(f"    channel std:   min={ch_std.min():.6f}   max={ch_std.max():.6f}")

    # Histogram
    print(f"\n  Histogram (20 bins):")
    counts, edges = np.histogram(fp, bins=20)
    max_count = counts.max()
    for i in range(len(counts)):
        bar_len = int(40 * counts[i] / max_count) if max_count > 0 else 0
        print(f"    [{edges[i]:>8.4f}, {edges[i+1]:>8.4f}) {counts[i]:>10d} {'█' * bar_len}")


def main():
    parser = argparse.ArgumentParser(description="Read layer outputs from run_backend.py")
    parser.add_argument("--npz", type=str, default=DEFAULT_NPZ, help="Path to layer_outputs.npz")
    parser.add_argument("--layer", type=str, default=None,
                        help="Show detail for a specific layer key (use __ for dots, e.g. backbone__4__1__rnn)")
    parser.add_argument("--list", action="store_true", help="List all available layer keys")
    args = parser.parse_args()

    assert os.path.exists(args.npz), f"File not found: {args.npz}"
    data = np.load(args.npz, allow_pickle=True)

    if args.list:
        print(f"Available layers ({len(data.files)}):")
        for key in data.files:
            print(f"  {key:<50}  ->  {key_to_name(key)}")
        return

    if args.layer:
        if args.layer not in data.files:
            # Try converting dots to __
            alt_key = args.layer.replace(".", "__")
            if alt_key in data.files:
                args.layer = alt_key
            else:
                print(f"Key '{args.layer}' not found. Available keys:")
                for k in data.files:
                    print(f"  {k}")
                return
        print_detail(data, args.layer)
        return

    # Default: summary
    print(f"[layer_outputs.npz] {len(data.files)} layers\n")
    print_summary(data)


if __name__ == "__main__":
    main()
