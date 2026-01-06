# Pruning Huada Directory - Migration Status

This directory contains the original pruning-related files before migration to the main codebase.

## Migration Completed ✓

The following files have been **successfully migrated** to their new locations:

### 1. `lstmIOquant_jjy.py` → `/workspace/huada/scall/quantization/`
- **Status**: Migrated and updated
- **Changes**: 
  - Added command-line argument parsing for configurable paths
  - Changed default paths to workspace-relative
  - Ready for use with: `python quantization/lstmIOquant_jjy.py --help`

### 2. `train_index_ddp_sparse.py` → `/workspace/huada/scall/opencall_cli/`
- **Status**: Migrated and updated
- **Changes**:
  - Fixed import: `from bonito.sparseoptimizer.pruning` → `from pruning`
  - Updated result directory: `/store/zjj/task_results/` → `/workspace/huada/task_results/`
  - Removed `model_name` parameter from PrunerScheduler initialization
  - Compatible with local `pruning` module

### 3. `train_index_ddp_sparse.sh` → `/workspace/huada/scall/opencall_cli/scripts/`
- **Status**: Migrated and modernized
- **Changes**:
  - Updated to use `torchrun` instead of deprecated `torch.distributed.launch`
  - Removed hardcoded `CUDA_VISIBLE_DEVICES`
  - Added dynamic port finding
  - Updated example paths to workspace-relative

## Files in This Directory

### Keep (Compiled Binaries)
- **`pruning_so/`**: Compiled `.so` files for pruning operations
  - These are pre-compiled binary files that may be used by the system
  - **Action**: Keep as reference or for compatibility

### Obsolete (Use Main Codebase Instead)
- **`rnn_ctc_crf/model.py`**: Custom model with hardcoded paths
  - **Issue**: Contains hardcoded `sys.path.append('/store/zjj/coding/tmp/opencall/opencall/libs')`
  - **Replacement**: Use `/workspace/huada/scall/opencall/models/rnn_ctc_crf/model.py` (has proper relative imports)
  - **Action**: Can be removed - use main opencall models

- **`rnn_ctc_crf/basecall.py`**: Inference-related basecalling code
  - **Usage**: Only for inference, not needed for training
  - **Action**: Can be removed or kept as reference

- **`train_index_ddp_sparse.py`**: Original training script (outdated)
  - **Action**: Can be removed - use `/workspace/huada/scall/opencall_cli/train_index_ddp_sparse.py`

- **`train_index_ddp_sparse.sh`**: Original shell script (outdated)
  - **Action**: Can be removed - use `/workspace/huada/scall/opencall_cli/scripts/train_index_ddp_sparse.sh`

- **`lstmIOquant_jjy.py`**: Original quantization script (outdated)
  - **Action**: Can be removed - use `/workspace/huada/scall/quantization/lstmIOquant_jjy.py`

## Usage of Migrated Files

### Running Sparse Training
```bash
cd /workspace/huada/scall/opencall_cli
bash scripts/train_index_ddp_sparse.sh 2 \
    /workspace/huada/train_data/train \
    /workspace/huada/models/weights_0.tar \
    64 0.0004 10000000000 0 3 lstm_ctc_crf 200 my_sparse_model
```

### Running Quantization
```bash
cd /workspace/huada/scall
python quantization/lstmIOquant_jjy.py \
    --config_file /path/to/config.toml \
    --pretrained_model /path/to/weights.tar \
    --data_dir /workspace/huada/scall/train_data \
    --act_scales_path /workspace/huada/ckpt/act_scales.pth \
    --io_quant_path /workspace/huada/ckpt/io_quant.pth
```

## Directory Cleanup Recommendation

To clean up obsolete files while keeping binaries:
```bash
cd /workspace/huada/scall/pruning_huada
# Remove obsolete Python files (migrated to new locations)
rm -f train_index_ddp_sparse.py train_index_ddp_sparse.sh lstmIOquant_jjy.py
# Remove obsolete model files (use main opencall models instead)
rm -rf rnn_ctc_crf/
# Keep pruning_so/ directory with compiled binaries
```

Or, to keep everything as reference:
- Keep this directory unchanged as a historical reference
- Use only the migrated files in their new locations

## Notes

- The local `pruning` module at `/workspace/huada/scall/pruning/` provides the `PrunerScheduler` class
- All imports should use `from pruning import PrunerScheduler` instead of bonito imports
- Result paths default to `/workspace/huada/task_results/` for consistency



