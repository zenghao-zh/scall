import numpy as np


def _dtype_key(dtype):
    if dtype is None:
        return ""
    return str(dtype).lower()


def _ceil_div(a, b):
    return (a + b - 1) // b


def _topk_mask_lastdim_abs(x, k):
    """
    Return bool mask of top-k by abs value along the last dimension.
    x: [..., cols]
    """
    cols = x.shape[-1]
    if k <= 0:
        return np.zeros_like(x, dtype=bool)
    if k >= cols:
        return np.ones_like(x, dtype=bool)

    idx = np.argpartition(np.abs(x), kth=cols - k, axis=-1)[..., -k:]
    mask = np.zeros_like(x, dtype=bool)
    np.put_along_axis(mask, idx, True, axis=-1)
    return mask


def _block_sparsity_balance_2d_fast(weight_io, keep_k, asic_input_gloup, group_size, block_size, patch_size):
    """
    Fast path for update_mask_asic_2d core.

    weight_io: [I, O]
    Requirements:
      - I % group_size == 0
      - group_size % block_size == 0
      - block_size % asic_input_gloup == 0
    """
    in_size, out_size = weight_io.shape
    num_groups = in_size // group_size
    lane = asic_input_gloup
    lane_len = block_size // lane

    # Original code:
    # keep_k0 = int(block_size * num_groups * keep_k / in_size)
    keep_k0 = int(block_size * num_groups * keep_k / in_size)

    # Original _block_1x1 allocates keep per lane as:
    # int(len(lane_ids) / total_cols * keep_k0), min 1
    # Here total_cols = num_groups * block_size, len(lane_ids)=num_groups*lane_len
    lane_keep = max(int((num_groups * lane_len) / (num_groups * block_size) * keep_k0), 1)

    # Original _block_sparsity_balance effectively uses ceil(lane_keep / out_size) per row
    row_k = min(num_groups * lane_len, _ceil_div(lane_keep, out_size))

    # [I, O] -> [G, P, B, O]
    x = np.ascontiguousarray(weight_io).reshape(num_groups, patch_size, block_size, out_size)

    # Inside each block, original code does interleaved grouping by idx % asic_input_gloup.
    # If block_size = lane_len * lane, then:
    # [G, P, B, O] -> [O, P, lane, G * lane_len]
    x = x.reshape(num_groups, patch_size, lane_len, lane, out_size)
    x = x.transpose(4, 1, 3, 0, 2).reshape(out_size, patch_size, lane, num_groups * lane_len)

    mask = _topk_mask_lastdim_abs(x, row_k)

    # Reverse:
    # [O, P, lane, G * lane_len] -> [I, O]
    mask = mask.reshape(out_size, patch_size, lane, num_groups, lane_len)
    mask = mask.transpose(3, 1, 4, 2, 0).reshape(in_size, out_size)
    return mask.astype(weight_io.dtype)


def _update_mask_asic_2d_single(weight, keep_k, dtype, asic_input_gloup=8, cgb=512, group_size_value=64):
    """
    Single-chunk version of update_mask_asic_2d.
    weight: [I, O]
    """
    dtype_key = _dtype_key(dtype)

    if keep_k < 1:
        return np.zeros_like(weight)

    in_size, out_size = weight.shape

    if dtype_key in ["bf16", "bfloat16"]:
        group_size = cgb
        block_size = group_size_value // 2
    elif dtype_key == "int8":
        group_size = cgb
        block_size = group_size_value
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    if in_size < block_size:
        block_size = in_size

    if group_size > block_size:
        assert group_size % block_size == 0

    patch_size = max(group_size // block_size, 1)

    # Original min-group fallback logic
    if dtype_key == "int8" and group_size < group_size_value:
        group_size = group_size_value
    elif dtype_key in ["bf16", "bfloat16"] and group_size < (group_size_value // 2):
        group_size = group_size_value // 2

    # Fast path: regular aligned case
    can_fast = (
        in_size > 0
        and group_size >= block_size
        and group_size % block_size == 0
        and in_size % group_size == 0
        and block_size % asic_input_gloup == 0
        and patch_size == (group_size // block_size)
    )

    assert can_fast, f"Not Implemented yet"

    return _block_sparsity_balance_2d_fast(
        weight_io=weight,
        keep_k=keep_k,
        asic_input_gloup=asic_input_gloup,
        group_size=group_size,
        block_size=block_size,
        patch_size=patch_size,
    )


def update_mask_asic_2d(weight, keep_k, dtype, asic_input_gloup=8, cgb=512, group_size_value=64):
    """
    weight: [I, O]
    """
    if keep_k < 1:
        return np.zeros_like(weight)

    dtype_key = _dtype_key(dtype)

    if dtype_key in ["bf16", "bfloat16"]:
        group_size_max = 256
    elif dtype_key == "int8":
        group_size_max = 512
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    in_size, out_size = weight.shape

    # Keep the original chunking policy
    # Original condition: (in_size / patch_size) > group_size_max
    if dtype_key in ["bf16", "bfloat16"]:
        block_size = group_size_value // 2
    else:
        block_size = group_size_value

    patch_size = max(cgb // block_size, 1)

    if (in_size / patch_size) > group_size_max:
        ori_in_size = in_size
        ori_keep_k = keep_k
        inc_group_size = int(np.ceil(in_size / group_size_max))
        temp_list = []

        for i in range(inc_group_size):
            weight_group = weight[i * group_size_max:(i + 1) * group_size_max, :]
            if weight_group.size == 0:
                continue
            group_keep = ori_keep_k * (weight_group.shape[0] / ori_in_size)
            mask_group = _update_mask_asic_2d_single(
                weight=weight_group,
                keep_k=group_keep,
                dtype=dtype,
                asic_input_gloup=asic_input_gloup,
                cgb=cgb,
                group_size_value=group_size_value,
            )
            temp_list.append(mask_group)

        if not temp_list:
            return np.zeros_like(weight)

        return np.concatenate(temp_list, axis=0).astype(weight.dtype)

    return _update_mask_asic_2d_single(
        weight=weight,
        keep_k=keep_k,
        dtype=dtype,
        asic_input_gloup=asic_input_gloup,
        cgb=cgb,
        group_size_value=group_size_value,
    )


def prune_dim_2(weight, keep_k, dtype_, cgb, group_size_value=64):
    """
    weight: [O, I]
    keep_k: float/int
    cgb: int
    dtype_: str
    """
    new_params = np.transpose(weight, [1, 0])  # [I, O]
    new_shape = new_params.shape

    if new_shape[0] > 2048:
        value = int(np.ceil(new_shape[0] / 2048))
        tmp = []
        for i in range(value):
            new_params_ = new_params[i * 2048:(i + 1) * 2048, :]
            if new_params_.size == 0:
                continue
            group_keep = keep_k * (new_params_.shape[0] / new_params.shape[0])
            tmp_group_mask = update_mask_asic_2d(
                weight=new_params_,
                keep_k=group_keep,
                dtype=dtype_,
                cgb=cgb,
                group_size_value=group_size_value
            )
            tmp.append(tmp_group_mask)
        mask = np.concatenate(tmp, axis=0)
        mask = np.transpose(mask, [1, 0])
    else:
        tmp_mask = update_mask_asic_2d(
            weight=new_params,
            keep_k=keep_k,
            dtype=dtype_,
            cgb=cgb,
            group_size_value=group_size_value
        )
        mask = np.transpose(tmp_mask, [1, 0])

    return mask.astype(weight.dtype)


def prune_func(weight, sparsity, dtype_info, cgb, group_size_value=64):
    keep_k = max(int(weight.size * (1.0 - sparsity)), 1)

    assert len(weight.shape) == 2
    mask = prune_dim_2(
        weight=weight,
        keep_k=keep_k,
        dtype_=dtype_info,
        cgb=cgb,
        group_size_value=group_size_value
    )
    new_params = weight * mask
    return new_params