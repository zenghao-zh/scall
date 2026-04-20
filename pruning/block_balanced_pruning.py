"""
为 ASIC 稀疏推理生成权重掩码 (PyTorch 全向量化实现)。
 
目标：给定 [I, O] 权重矩阵和要保留的非零个数 keep_k，生成 0/1 掩码，
使非零权重在各 output channel 和各硬件子组间均匀分布。
 
原理
----
ASIC 对 I 维有三级硬件分区约束：
 
  I 维全部行
    ├─ 段 (segment)      : I 过大时按 group_size_max 切段
    │   ├─ 条带 (patch)  : 每段内按 block_size 切块，跨 CG 组同位置块组成一个 patch
    │   │   ├─ 子组 0    : patch 内按 i%G 交错，G=8 路对应硬件并行通道
    │   │   ├─ 子组 1
    │   │   └─ ...
    │   └─ ...
    └─ ...
 
分区完成后，每个叶子组(段×patch×子组)内做相同的 per-channel balanced topk：
每个 output channel 保留 ceil(keep_k_leaf / O) 个绝对值最大的权重。
 
整条流水线只需一次 topk，全程 reshape 零拷贝。
"""
 
import math
import torch
 
# 不同数据类型的硬件约束
#   group_size_max : 单段 I 维上限
#   block_ratio    : block_size = group_size_value × block_ratio
DTYPE_PARAMS = {
    "int8":     {"group_size_max": 512, "block_ratio": 1},
    "bf16":     {"group_size_max": 256, "block_ratio": 0.5},
    "bfloat16": {"group_size_max": 256, "block_ratio": 0.5},
}
 
 
def update_mask_asic_2d(
    weight: torch.Tensor,
    keep_k: int,
    dtype: str,
    asic_input_gloup: int = 8,
    cgb: int = 512,
    group_size_value: int = 64,
) -> torch.Tensor:
    """为 [I, O] 权重矩阵生成 block-balanced 稀疏掩码。
 
    Args:
        weight:           权重矩阵 [I, O]
        keep_k:           需保留的非零权重总数
        dtype:            权重数据类型 ("int8" / "bf16" / "bfloat16")
        asic_input_gloup: 硬件交错通道数 (默认 8)
        cgb:              channel group boundary (默认 512)
        group_size_value: 基础 group size (默认 64)
 
    Returns:
        mask: 0/1 掩码 [I, O]，与 weight 同设备同 dtype
    """
    if keep_k < 1:
        return torch.zeros_like(weight)
 
    I, O = weight.shape
    dev = weight.device
    G = asic_input_gloup
 
    # ════════════════ 硬件参数 ════════════════
    cfg = DTYPE_PARAMS[dtype]
    group_size_max = cfg["group_size_max"]
    block_size = min(int(group_size_value * cfg["block_ratio"]), I)
    group_size = max(cgb, block_size)
    assert group_size >= block_size and group_size % block_size == 0
 
    P = max(group_size // block_size, 1)            # patch 数 (每 CG 组内的 block 数)
 
    # ════════════════ 第 1 级：分段 ════════════════
    # I 过大时切成等长段，pad 末尾使整除
    need_seg = (I / P > group_size_max)
    seg_size = group_size_max if need_seg else I
    S = math.ceil(I / seg_size) if need_seg else 1  # 段数
    pad_I = S * seg_size
 
    w = weight
    if pad_I > I:
        w = weight.new_zeros(pad_I, O)
        w[:I] = weight
    w = w.reshape(S, seg_size, O)                   # [S, seg_size, O]
 
    # ════════════════ 第 2 级：构建 patch 行索引 ════════════════
    # 每段内有 num_cg 个 CG 组，每组 group_size 行
    # patch p 从每个 CG 组取第 p 个 block (block_size 行)
    # 例: seg=1024, gs=512, bs=64 → 2 个 CG 组, 8 个 patch
    #   patch 0 = [0..63, 512..575],  patch 1 = [64..127, 576..639], ...
    num_cg = max(math.ceil(seg_size / group_size), 1)
    L = num_cg * block_size                         # 每个 patch 的行数
 
    cg_starts = torch.arange(num_cg, device=dev) * group_size
    blk_idx = torch.arange(block_size, device=dev)
    base = (cg_starts[:, None] + blk_idx[None, :]).reshape(-1)       # [L] patch 0 的行号
 
    patch_offsets = torch.arange(P, device=dev) * block_size
    all_rows = base[None, :] + patch_offsets[:, None]                 # [P, L] 所有 patch 的行号
 
    valid = all_rows < seg_size                                       # [P, L]
    rows_c = all_rows.clamp(max=max(seg_size - 1, 0))
 
    # ════════════════ gather 权重 ════════════════
    idx = rows_c[None, :, :, None].expand(S, P, L, O)
    pw = torch.gather(w[:, None].expand(-1, P, -1, -1), 2, idx)      # [S, P, L, O]
    pw = pw * valid[None, :, :, None].to(pw.dtype)                    # 越界位置置零
 
    # ════════════════ 第 3 级：交错分组 + topk ════════════════
    # 将每个 patch 的 L 行按 i%G 分成 G 个子组，每组 rpg 行
    rpg = math.ceil(L / G)                          # rows per interleaved group
    pad_L = rpg * G
    if pad_L > L:
        pw2 = pw.new_zeros(S, P, pad_L, O)
        pw2[:, :, :L] = pw
    else:
        pw2 = pw
 
    # 交错重排 → 展平为 topk 输入
    #   [S, P, pad_L, O]
    #   → [S, P, rpg, G, O]     按 i%G 拆出子组维度
    #   → [S*P*G, O, rpg]       合并 batch，转置为 [batch, O, rpg] 以便 per-channel topk
    flat = (pw2.reshape(S, P, rpg, G, O)
               .permute(0, 1, 3, 4, 2)
               .reshape(S * P * G, O, rpg))
 
    # 计算每个 output channel 保留几个权重
    seg_k = keep_k * (seg_size / I) if need_seg else keep_k
    patch_keep = int(L * seg_k / seg_size)
    gk = max(patch_keep // G, 1)
    per_row = min(math.ceil(gk / O), rpg) if O > 0 else 0
 
    # 快速路径
    if per_row <= 0:
        return torch.zeros(I, O, device=dev, dtype=weight.dtype)
    if per_row >= rpg:
        return torch.ones(I, O, device=dev, dtype=weight.dtype)
 
    # 单次 topk：所有段×patch×子组 一起处理
    _, topk_idx = torch.topk(flat.abs(), per_row, dim=-1)             # [S*P*G, O, per_row]
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(-1, topk_idx, 1.0)
 
    # ════════════════ 逆变换 + scatter 写回 ════════════════
    # topk 结果还原到 patch 行顺序
    #   [S*P*G, O, rpg] → [S, P, G, O, rpg] → [S, P, rpg, G, O] → [S, P, pad_L, O]
    mask_pw = (mask_flat
               .reshape(S, P, G, O, rpg)
               .permute(0, 1, 4, 2, 3)
               .reshape(S, P, pad_L, O))
    if pad_L > L:
        mask_pw = mask_pw[:, :, :L]
 
    # 将各 patch 的 mask 写回段级 mask
    flat_valid = valid.reshape(-1)                                    # [P*L]
    v_rows = rows_c.reshape(-1)[flat_valid]                           # [num_valid]
    v_mask = mask_pw.reshape(S, P * L, O)[:, flat_valid]              # [S, num_valid, O]
 
    seg_mask = torch.ones(S, seg_size, O, device=dev, dtype=weight.dtype)
    seg_mask.scatter_(1, v_rows[None, :, None].expand(S, -1, O), v_mask.to(weight.dtype))
 
    return seg_mask.reshape(pad_I, O)[:I]