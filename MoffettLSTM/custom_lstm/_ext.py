from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


@lru_cache(maxsize=1)
def load_extension():
    root = Path(__file__).resolve().parents[1]
    sources = [str(root / "csrc" / "lstm_cpu.cpp")]
    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3", "-std=c++17"]
    extra_include_paths = [str(root / "third_party" / "moffett_pytorch_sigmoid_tanh_cuda_v3")]
    if torch.version.cuda is not None:
        sources.append(str(root / "csrc" / "lstm_cuda.cu"))
        extra_cflags.append("-DWITH_CUDA")
    build_dir = root / ".build"
    build_dir.mkdir(exist_ok=True)
    return load(
        name="fast_lstm_ext",
        sources=sources,
        build_directory=str(build_dir),
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        verbose=False,
    )
