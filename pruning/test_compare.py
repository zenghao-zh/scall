import numpy as np
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import compiler_frontend_prune_unit as np_mod
import compiler_frontend_prune_unit_torch as torch_mod

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"CUDA device: {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else "Running on CPU only")
print()


def compare(weight_np, sparsity, dtype_info, cgb, group_size_value=64, label=""):
    weight_cpu = torch.from_numpy(weight_np.copy())
    weight_gpu = weight_cpu.to(DEVICE)

    try:
        result_np = np_mod.prune_func(weight_np, sparsity, dtype_info, cgb, group_size_value)
    except Exception as e:
        print(f"[ERROR-NP]    {label}: {e}")
        return None
    try:
        result_gpu = torch_mod.prune_func(weight_gpu, sparsity, dtype_info, cgb, group_size_value)
    except Exception as e:
        print(f"[ERROR-TORCH] {label}: {e}")
        return None

    result_gpu_np = result_gpu.cpu().numpy()

    match = np.array_equal(result_np, result_gpu_np)
    max_diff = np.max(np.abs(result_np - result_gpu_np)) if not match else 0.0
    nonzero_np = np.count_nonzero(result_np)
    nonzero_gpu = np.count_nonzero(result_gpu_np)
    mask_match = np.array_equal(result_np != 0, result_gpu_np != 0)

    status = "PASS" if match else "FAIL"
    print(f"[{status}] {label}")
    print(f"  shape={weight_np.shape}, sparsity={sparsity}, dtype={dtype_info}, cgb={cgb}, gs={group_size_value}")
    if not match:
        print(f"  max_diff={max_diff:.6e}")
        print(f"  nonzero: np={nonzero_np}, torch_cuda={nonzero_gpu}")
        print(f"  mask_match (same zero pattern)={mask_match}")
    return match


np.random.seed(42)

passed = 0
failed = 0
errored = 0

test_cases = [
    # --- int8 ---
    ((512, 512), 0.5, "int8", 512, 64, "int8 512x512 sp=0.5"),
    ((512, 512), 0.3, "int8", 512, 64, "int8 512x512 sp=0.3"),
    ((512, 512), 0.7, "int8", 512, 64, "int8 512x512 sp=0.7"),
    ((512, 512), 0.9, "int8", 512, 64, "int8 512x512 sp=0.9"),
    ((512, 512), 0.1, "int8", 512, 64, "int8 512x512 sp=0.1"),
    ((256, 512), 0.5, "int8", 512, 64, "int8 256x512 sp=0.5"),
    ((128, 512), 0.5, "int8", 512, 64, "int8 128x512 sp=0.5"),
    ((1024, 512), 0.5, "int8", 512, 64, "int8 1024x512 sp=0.5"),
    ((1024, 1024), 0.5, "int8", 512, 64, "int8 1024x1024 sp=0.5"),
    ((64, 1024), 0.5, "int8", 512, 64, "int8 64x1024 sp=0.5"),
    ((512, 2048), 0.5, "int8", 512, 64, "int8 512x2048 sp=0.5"),

    # --- bf16 ---
    ((512, 512), 0.5, "bf16", 512, 64, "bf16 512x512 sp=0.5"),
    ((512, 512), 0.3, "bf16", 512, 64, "bf16 512x512 sp=0.3"),
    ((512, 512), 0.7, "bf16", 512, 64, "bf16 512x512 sp=0.7"),
    ((256, 512), 0.5, "bf16", 512, 64, "bf16 256x512 sp=0.5"),
    ((128, 512), 0.5, "bf16", 512, 64, "bf16 128x512 sp=0.5"),
    ((1024, 512), 0.5, "bf16", 512, 64, "bf16 1024x512 sp=0.5"),
    ((1024, 1024), 0.5, "bf16", 512, 64, "bf16 1024x1024 sp=0.5"),

    # --- >2048 path ---
    ((256, 4096), 0.5, "int8", 512, 64, "int8 256x4096 >2048 sp=0.5"),
    ((256, 4096), 0.5, "bf16", 512, 64, "bf16 256x4096 >2048 sp=0.5"),
    ((512, 3072), 0.6, "int8", 512, 64, "int8 512x3072 >2048 sp=0.6"),
    ((512, 8192), 0.5, "int8", 512, 64, "int8 512x8192 >2048 sp=0.5"),
    ((1024, 4096), 0.4, "bf16", 512, 64, "bf16 1024x4096 >2048 sp=0.4"),

    # --- Different cgb ---
    ((512, 256), 0.5, "int8", 256, 64, "int8 cgb=256 512x256"),
    ((512, 256), 0.5, "bf16", 256, 64, "bf16 cgb=256 512x256"),
    ((256, 256), 0.5, "int8", 256, 64, "int8 cgb=256 256x256"),

    # --- Different group_size_value ---
    ((512, 512), 0.5, "int8", 512, 128, "int8 gs=128 512x512"),
    ((512, 512), 0.5, "bf16", 512, 128, "bf16 gs=128 512x512"),

    # --- Larger ---
    ((2048, 1024), 0.5, "int8", 512, 64, "int8 2048x1024 sp=0.5"),
    ((2048, 1024), 0.5, "bf16", 512, 64, "bf16 2048x1024 sp=0.5"),
    ((512, 512), 0.45, "int8", 512, 64, "int8 512x512 sp=0.45"),
    ((512, 512), 0.63, "bf16", 512, 64, "bf16 512x512 sp=0.63"),
]

print("=" * 70)
print("Correctness: NumPy (CPU) vs PyTorch (CUDA)")
print("=" * 70)
print()

for shape, sparsity, dtype, cgb, gs, label in test_cases:
    weight = np.random.randn(*shape).astype(np.float32)
    result = compare(weight, sparsity, dtype, cgb, gs, label)
    if result is True:
        passed += 1
    elif result is False:
        failed += 1
    else:
        errored += 1

print()
print("=" * 70)
print(f"Summary: {passed} PASS, {failed} FAIL, {errored} ERROR")
print("=" * 70)

# --- Performance: NumPy vs Torch-CPU vs Torch-CUDA ---
print()
print("=" * 70)
print("Performance: NumPy vs Torch-CPU vs Torch-CUDA  (avg of 20 runs)")
print("=" * 70)
print()

perf_cases = [
    ((512, 512), 0.5, "int8", 512, 64, "int8 512x512"),
    ((1024, 1024), 0.5, "int8", 512, 64, "int8 1024x1024"),
    ((256, 4096), 0.5, "int8", 512, 64, "int8 256x4096"),
    ((512, 8192), 0.5, "bf16", 512, 64, "bf16 512x8192"),
    ((2048, 1024), 0.5, "int8", 512, 64, "int8 2048x1024"),
    ((2048, 2048), 0.5, "int8", 512, 64, "int8 2048x2048"),
    ((1024, 4096), 0.5, "bf16", 512, 64, "bf16 1024x4096"),
]

N_RUNS = 20

header = f"{'Case':25s}  {'NumPy':>9s}  {'Torch-CPU':>10s}  {'Torch-CUDA':>11s}  {'CPU speedup':>11s}  {'CUDA speedup':>12s}"
print(header)
print("-" * len(header))

for shape, sparsity, dtype, cgb, gs, label in perf_cases:
    weight_np = np.random.randn(*shape).astype(np.float32)
    weight_cpu = torch.from_numpy(weight_np.copy())
    weight_gpu = weight_cpu.to(DEVICE)

    # warmup
    np_mod.prune_func(weight_np, sparsity, dtype, cgb, gs)
    torch_mod.prune_func(weight_cpu, sparsity, dtype, cgb, gs)
    torch_mod.prune_func(weight_gpu, sparsity, dtype, cgb, gs)
    torch.cuda.synchronize()

    # numpy
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        np_mod.prune_func(weight_np, sparsity, dtype, cgb, gs)
    np_time = (time.perf_counter() - t0) / N_RUNS * 1000

    # torch cpu
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        torch_mod.prune_func(weight_cpu, sparsity, dtype, cgb, gs)
    torch_cpu_time = (time.perf_counter() - t0) / N_RUNS * 1000

    # torch cuda (with sync)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_RUNS):
        torch_mod.prune_func(weight_gpu, sparsity, dtype, cgb, gs)
        torch.cuda.synchronize()
    torch_gpu_time = (time.perf_counter() - t0) / N_RUNS * 1000

    cpu_speedup = np_time / torch_cpu_time if torch_cpu_time > 0 else float('inf')
    gpu_speedup = np_time / torch_gpu_time if torch_gpu_time > 0 else float('inf')

    print(f"{label:25s}  {np_time:8.2f}ms  {torch_cpu_time:9.2f}ms  {torch_gpu_time:10.2f}ms  {cpu_speedup:10.1f}x  {gpu_speedup:11.1f}x")
