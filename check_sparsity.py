"""Check per-layer sparsity of a checkpoint.

Usage:
    python check_sparsity.py /workspace/huada/task_results/lstm_ctc_crf_qat_int8/weights_0.tar
"""

import sys
import torch


def check_sparsity(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    print(f"Checkpoint: {ckpt_path}")
    print(f"Total layers: {len(state_dict)}")
    print(f"{'Layer':<50} {'Shape':<25} {'Total':>10} {'Zeros':>10} {'Sparsity':>10}")
    print("-" * 110)

    total_params = 0
    total_zeros = 0

    for name, param in state_dict.items():
        numel = param.numel()
        zeros = (param == 0).sum().item()
        sparsity = zeros / numel * 100 if numel > 0 else 0.0
        total_params += numel
        total_zeros += zeros

        shape_str = str(tuple(param.shape))
        print(f"{name:<50} {shape_str:<25} {numel:>10} {zeros:>10} {sparsity:>9.2f}%")

    overall = total_zeros / total_params * 100 if total_params > 0 else 0.0
    print("-" * 110)
    print(f"{'OVERALL':<50} {'':<25} {total_params:>10} {total_zeros:>10} {overall:>9.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <checkpoint_path>")
        sys.exit(1)
    check_sparsity(sys.argv[1])
