"""
验证权重是否满足 CGB32 / CGB256 稀疏格式。

CGB 稀疏的核心约束：
  权重按列（输入通道）分组，每组内，每个输出通道保留的非零元素数量相同（±1），即"平衡稀疏"。

具体分组流程（以 CGB32, bf16, group_size_value=64 为例）：
  1. 权重 [O, I] 转置为 [I, O]
  2. 若 I > 2048，先拆成 2048 大小的块
  3. 每个块再拆成 group_size_max (256 for bf16) 大小的段
  4. 每段内的通道按 stride=asic_input_gloup(8) 交错分成 8 个子组
  5. 每个子组约 32 个通道，要求每个输出通道的非零元素数相同 (±1)

Usage:
    python verify_cgb_sparsity.py --weight_path <path> [--key <param_name>] [--cgb 32] [--dtype bf16]
    
    或在 Python 中:
        from verify_cgb_sparsity import verify_cgb_sparsity
        verify_cgb_sparsity(weight_tensor, cgb=32, dtype='bf16')
"""

import numpy as np
import argparse


def verify_cgb_sparsity(weight, cgb=32, dtype='bf16', group_size_value=64, verbose=True):
    """
    验证 2D 权重是否满足 CGB 稀疏格式约束。
    
    Args:
        weight: [O, I] 的 numpy array 或 torch.Tensor
        cgb: CGB 值 (32, 64, 256, 512 等)
        dtype: 'bf16' 或 'int8'
        group_size_value: 默认 64
        verbose: 是否打印详细信息
    
    Returns:
        dict: 包含验证结果的字典
    """
    try:
        import torch
        if torch.is_tensor(weight):
            weight = weight.detach().cpu().float().numpy()
    except ImportError:
        pass
    
    weight = np.array(weight, dtype=np.float32)
    
    if weight.ndim != 2:
        raise ValueError(f"仅支持 2D 权重，当前维度: {weight.ndim}")
    
    O, I = weight.shape
    
    # ========== 基础信息 ==========
    total_elements = weight.size
    zero_elements = np.sum(weight == 0)
    nonzero_elements = total_elements - zero_elements
    sparsity = zero_elements / total_elements
    
    if verbose:
        print("=" * 70)
        print(f"权重形状: ({O}, {I})")
        print(f"总体稀疏率: {sparsity:.4f} ({zero_elements}/{total_elements} 个零)")
        print(f"非零元素数: {nonzero_elements}")
        print(f"CGB: {cgb}, dtype: {dtype}, group_size_value: {group_size_value}")
        print("=" * 70)
    
    if sparsity < 0.01:
        if verbose:
            print("⚠️  权重几乎没有稀疏（稀疏率 < 1%），无需验证 CGB 格式。")
        return {'valid': True, 'sparsity': sparsity, 'violations': [], 'message': 'not sparse'}
    
    # ========== 参数计算（与 pruning 代码一致）==========
    if dtype in ['bf16', 'bfloat16']:
        group_size_max = 256
        block_size = group_size_value // 2   # 32
    elif dtype == 'int8':
        group_size_max = 512
        block_size = group_size_value         # 64
    else:
        raise ValueError(f"不支持的 dtype: {dtype}")
    
    group_size = cgb
    asic_input_gloup = 512 // group_size_value  # 通常是 8
    patch_size = max(group_size // block_size, 1)
    
    if verbose:
        print(f"\n参数: group_size={group_size}, block_size={block_size}, "
              f"group_size_max={group_size_max}, asic_input_gloup={asic_input_gloup}, "
              f"patch_size={patch_size}")
    
    # ========== 转置为 [I, O]（与 prune_dim_2 一致）==========
    wt = weight.T  # [I, O]
    
    violations = []
    total_checks = 0
    
    # Step 1: 按 2048 拆块（prune_dim_2 中的逻辑）
    chunk_size = 2048
    if I > chunk_size:
        num_chunks = int(np.ceil(I / chunk_size))
    else:
        num_chunks = 1
        chunk_size = I
    
    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * 2048 if I > 2048 else 0
        c_end = min(c_start + 2048, I)
        chunk = wt[c_start:c_end, :]  # [chunk_I, O]
        chunk_I = chunk.shape[0]
        
        # Step 2: 按 group_size_max 拆段
        use_big_split = (chunk_I / patch_size) > group_size_max
        
        if use_big_split:
            num_groups = int(np.ceil(chunk_I / group_size_max))
        else:
            num_groups = 1
        
        for g_idx in range(num_groups):
            if use_big_split:
                g_start = g_idx * group_size_max
                g_end = min((g_idx + 1) * group_size_max, chunk_I)
            else:
                g_start = 0
                g_end = chunk_I
            
            group = chunk[g_start:g_end, :]  # [seg_I, O]
            seg_I = group.shape[0]
            
            # Step 3: 交错分成 asic_input_gloup 个子组
            for sg_idx in range(asic_input_gloup):
                indices = list(range(sg_idx, seg_I, asic_input_gloup))
                if not indices:
                    continue
                
                sub_group = group[indices, :]  # [sub_I, O]
                total_checks += 1
                
                # Step 4: 检查平衡性——每个输出通道的非零数应相同 (±1)
                nnz_per_output = np.sum(sub_group != 0, axis=0)  # [O]
                min_nnz = int(nnz_per_output.min())
                max_nnz = int(nnz_per_output.max())
                
                if max_nnz - min_nnz > 1:
                    abs_start = c_start + g_start
                    violations.append({
                        'chunk': chunk_idx,
                        'group': g_idx,
                        'sub_group': sg_idx,
                        'abs_input_start': abs_start,
                        'sub_channels': len(indices),
                        'min_nnz': min_nnz,
                        'max_nnz': max_nnz,
                        'diff': max_nnz - min_nnz,
                        'nnz_distribution': np.bincount(nnz_per_output.astype(int)).tolist(),
                    })
    
    # ========== 输出结果 ==========
    if verbose:
        print(f"\n共检查了 {total_checks} 个子组 "
              f"({num_chunks} chunks × {num_groups} groups × {asic_input_gloup} sub-groups)")
        
        if violations:
            print(f"\n❌ 发现 {len(violations)} 个不满足 CGB{cgb} 平衡约束的子组:")
            for i, v in enumerate(violations[:15]):
                print(f"  [{i+1}] Chunk {v['chunk']}, Group {v['group']}, "
                      f"Sub-group {v['sub_group']} "
                      f"(起始输入通道: {v['abs_input_start']}, "
                      f"{v['sub_channels']} 个通道): "
                      f"每输出通道 nnz: min={v['min_nnz']}, max={v['max_nnz']}, "
                      f"差值={v['diff']}")
                print(f"       nnz 分布(nnz值: 出现次数): {v['nnz_distribution']}")
            if len(violations) > 15:
                print(f"  ... 还有 {len(violations) - 15} 个违例")
        else:
            print(f"\n✅ 权重满足 CGB{cgb} ({dtype}) 平衡稀疏约束！")
        
        # 额外：打印每层的稀疏分布
        print(f"\n--- 按 group_size_max={group_size_max} 分段的稀疏率 ---")
        wt = weight.T
        seg_size = group_size_max
        num_segs = int(np.ceil(I / seg_size))
        for s in range(min(num_segs, 20)):
            seg = wt[s * seg_size: (s + 1) * seg_size, :]
            seg_sparsity = 1.0 - np.count_nonzero(seg) / seg.size
            bar = "█" * int(seg_sparsity * 40)
            print(f"  输入通道 [{s*seg_size:5d}:{min((s+1)*seg_size, I):5d}] "
                  f"稀疏率: {seg_sparsity:.4f} {bar}")
        if num_segs > 20:
            print(f"  ... 共 {num_segs} 段")
    
    result = {
        'valid': len(violations) == 0,
        'sparsity': sparsity,
        'num_violations': len(violations),
        'total_checks': total_checks,
        'violations': violations,
    }
    return result


def verify_model_cgb_sparsity(model_or_state_dict, verbose=True):
    """
    验证整个模型（或 state_dict）中所有 Linear 层的 CGB 稀疏格式。
    自动根据 pruning/misc.py 的规则确定每层的 cgb 和 dtype。
    
    Args:
        model_or_state_dict: PyTorch 模型或 state_dict
        verbose: 是否打印详细信息
    """
    import torch
    
    if isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        state_dict = model_or_state_dict.state_dict()
    
    results = {}
    
    for name, param in state_dict.items():
        if 'weight' not in name:
            continue
        if isinstance(param, torch.Tensor):
            w = param.detach().cpu().float().numpy()
        else:
            w = np.array(param, dtype=np.float32)
        
        if w.ndim != 2:
            continue
        
        shape = w.shape
        sparsity = 1.0 - np.count_nonzero(w) / w.size
        
        if sparsity < 0.01:
            continue  # 跳过非稀疏层
        
        # 根据 pruning/misc.py 的规则确定 cgb 和 dtype
        if 768 in shape:
            if 4096 in shape:
                cgb, dtype_str = 32, 'bf16'
            else:
                cgb, dtype_str = 256, 'bf16'
        else:
            cgb, dtype_str = 512, 'int8'
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"层: {name}  shape={shape}  → CGB{cgb} ({dtype_str})")
        
        result = verify_cgb_sparsity(w, cgb=cgb, dtype=dtype_str, verbose=verbose)
        results[name] = {
            'cgb': cgb,
            'dtype': dtype_str,
            **result
        }
    
    # 汇总
    if verbose:
        print(f"\n{'='*70}")
        print("汇总:")
        all_valid = True
        for name, r in results.items():
            status = "✅" if r['valid'] else "❌"
            print(f"  {status} {name:50s} CGB{r['cgb']:>3d} ({r['dtype']}) "
                  f"稀疏率={r['sparsity']:.4f} 违例={r['num_violations']}")
            if not r['valid']:
                all_valid = False
        
        if all_valid:
            print("\n✅ 所有稀疏层都满足对应的 CGB 约束！")
        else:
            print("\n❌ 存在不满足 CGB 约束的层，请检查上面的详细输出。")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="验证权重是否满足 CGB 稀疏格式")
    parser.add_argument('--weight_path', type=str, required=True, 
                        help='权重文件路径 (.pt / .pth / .npy)')
    parser.add_argument('--key', type=str, default=None,
                        help='state_dict 中的参数名（如不指定则检查全部 Linear 层）')
    parser.add_argument('--cgb', type=int, default=None,
                        help='CGB 值 (32/64/256/512)，不指定则自动判断')
    parser.add_argument('--dtype', type=str, default=None, choices=['bf16', 'int8'],
                        help='数据类型，不指定则自动判断')
    parser.add_argument('--group_size_value', type=int, default=64)
    args = parser.parse_args()
    
    import torch
    
    if args.weight_path.endswith('.npy'):
        weight = np.load(args.weight_path)
        if args.cgb is None or args.dtype is None:
            parser.error("加载 .npy 文件时必须手动指定 --cgb 和 --dtype")
        verify_cgb_sparsity(weight, cgb=args.cgb, dtype=args.dtype, 
                           group_size_value=args.group_size_value)
    else:
        data = torch.load(args.weight_path, map_location='cpu')
        
        if isinstance(data, dict) and args.key:
            # 指定了具体的 key
            if args.key in data:
                weight = data[args.key]
            elif 'state_dict' in data and args.key in data['state_dict']:
                weight = data['state_dict'][args.key]
            else:
                # 尝试作为模型的 state_dict
                print(f"找不到 key '{args.key}'，可用的 keys:")
                for k in (data.get('state_dict', data)).keys():
                    print(f"  {k}")
                exit(1)
            
            if torch.is_tensor(weight):
                weight = weight.float().numpy()
            
            cgb = args.cgb
            dtype_str = args.dtype
            if cgb is None or dtype_str is None:
                shape = weight.shape
                if 768 in shape and 4096 in shape:
                    cgb, dtype_str = 32, 'bf16'
                elif 768 in shape:
                    cgb, dtype_str = 256, 'bf16'
                else:
                    cgb, dtype_str = 512, 'int8'
                print(f"自动判断: CGB{cgb}, dtype={dtype_str}")
            
            verify_cgb_sparsity(weight, cgb=cgb, dtype=dtype_str,
                               group_size_value=args.group_size_value)
        else:
            # 检查整个模型
            if 'state_dict' in data:
                state_dict = data['state_dict']
            elif isinstance(data, dict):
                state_dict = data
            else:
                state_dict = data.state_dict()
            
            verify_model_cgb_sparsity(state_dict)
