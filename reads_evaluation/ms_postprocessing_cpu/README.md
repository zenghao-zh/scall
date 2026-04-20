# ms_postprocessing_cpu

CPU port of ms_postprocessing. No CUDA/GPU required.

## Build

```bash
cd ms_postprocessing_cpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Test (same data & flow as original CUDA version)

```bash
# parse_dat: read binary BufferChunk .dat + qscore .npy → output CSV
./Release/parse_dat 0 \
  ../ms_postprocessing/data/wy_bufferChunks_5.dat \
  ../ms_postprocessing/data/wy_qscore_5.npy \
  100 40 40 \
  output/wy_qscore_5_cpu.csv 1

# Compare with GPU ground-truth
diff output/wy_qscore_5_cpu.csv ../ms_postprocessing/data/gt.csv
# Expected: identical output (pure float arithmetic, same operation order)
```

## Changes from CUDA version

- `kernel.cu` → `kernel_cpu.cpp` (plain C++ loops + OpenMP)
- `postprocess.cpp/h` → removed `cudaMalloc`/`cudaFree`/`cudaMemcpy`
- `npy.h` → CPU-only (removed `cuda_runtime_api.h` and device-memory paths)
- `stress.cpp`/`demo.cpp` → `cudaEvent` timing replaced with `std::chrono`
- `gpu_id` parameter accepted for API compatibility but ignored
