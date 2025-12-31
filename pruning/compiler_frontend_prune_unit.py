import numpy as np


def np_topk(weight, k, axis=1):
    """
    perform topK based on np.argsort
    :param matrix: to be sorted
    :param K: select and sort the top K items
    :param axis: dimension to be sorted.
    :return:
    """
    full_sort = np.argsort(weight, axis=axis)[:, ::-1]
    return full_sort.take(np.arange(k), axis=axis)


def update_mask(weight, keep_k):
    if keep_k >= 1:
        reshape_weight = np.reshape(weight, -1)
        index = np_topk(np.abs(reshape_weight), keep_k)
        mask = np.zeros(reshape_weight.shape)
        mask[index] = 1
        mask = mask.reshape(weight.shape)
        mask = mask.astype(weight.dtype)
    else:
        mask = np.zeros_like(weight)
    return mask


def update_mask_asic_4d(weight, keep_k, asic_input_gloup=8, dtype='int8', cgb=512, group_size_value=64):
    def _block_sparsity_balance(transpose_weight, keep_k):
        reshape_weight = np.reshape(transpose_weight, [-1, transpose_weight.shape[-1]])
        base_k = keep_k // reshape_weight.shape[0]
        remain_k = keep_k % reshape_weight.shape[0]
        if remain_k > 0:
            index = np_topk(np.abs(reshape_weight), min(reshape_weight.shape[-1], base_k + 1))
        else:
            index = np_topk(np.abs(reshape_weight), min(reshape_weight.shape[-1], base_k))
        dim1 = []
        dim2 = []
        for i, temp in enumerate(index.tolist()):
            for j in temp:
                dim1.append(i)
                dim2.append(j)
        mask = np.zeros(reshape_weight.shape)
        mask[dim1, dim2] = 1
        mask = mask.reshape(transpose_weight.shape)
        mask = mask.transpose([1, 2, 3, 0])
        mask = mask.astype(dtype=transpose_weight.dtype)
        return mask

    if keep_k >= 1:
        h, w, i, o = weight.shape
        transpose_weight = np.transpose(weight, [3, 0, 1, 2])
        if transpose_weight.shape[1] == 1 and transpose_weight.shape[2] == 1:
            transpose_weight = np.squeeze(transpose_weight)
            transpose_weight = np.transpose(transpose_weight, [1, 0])
            mask = update_mask_asic_2d(transpose_weight, keep_k, dtype, asic_input_gloup, cgb,
                                       group_size_value=group_size_value)
            mask = np.reshape(mask, [h, w, i, o])
        else:
            group_size = None
            if dtype == 'int8':
                group_size = group_size_value
            elif dtype in ['bf16', 'bfloat16']:
                group_size = group_size_value // 2
            temp1 = transpose_weight.shape[-1] // group_size
            temp2 = transpose_weight.shape[-1] % group_size
            keep_k_1 = int(keep_k * temp1 * group_size / transpose_weight.shape[-1])
            keep_k_2 = keep_k - keep_k_1
            mask = np.ones(weight.shape)
            if temp1 > 0:
                for i in range(temp1):
                    transpose_weight_1 = transpose_weight[:, :, :, i * group_size: (i + 1) * group_size]
                    mask_1 = _block_sparsity_balance(transpose_weight_1, keep_k_1 // temp1)
                    mask[:, :, i * group_size: (i + 1) * group_size, :] = mask_1
            if temp2 > 0:
                transpose_weight_2 = transpose_weight[:, :, :, temp1 * group_size:]
                mask_2 = _block_sparsity_balance(transpose_weight_2, keep_k_2)
                mask[:, :, temp1 * group_size:, :] = mask_2
            mask = mask.astype(transpose_weight.dtype)
    else:
        mask = np.zeros_like(weight)
    return mask


def update_mask_asic_2d(weight, keep_k, dtype, asic_input_gloup=8, cgb=512, group_size_value=64):
    def _block_sparsity_balance(transpose_weight, keep_k):
        reshape_weight = transpose_weight
        base_k = keep_k // reshape_weight.shape[0]   # OI
        remain_k = keep_k % reshape_weight.shape[0]
        if remain_k > 0:
            index = np_topk(np.abs(reshape_weight), min(reshape_weight.shape[-1], base_k + 1))
        else:
            index = np_topk(np.abs(reshape_weight), min(reshape_weight.shape[-1], base_k))
        dim1 = []
        dim2 = []
        for i, temp in enumerate(index.tolist()):
            for j in temp:
                dim1.append(i)
                dim2.append(j)
        mask = np.zeros(reshape_weight.shape)
        mask[dim1, dim2] = 1
        mask = mask.transpose([1, 0])
        mask = mask.astype(dtype=transpose_weight.dtype)
        return mask

    def _block_1x1(transpose_weight, keep_k, asic_input_gloup=8):
        '''
        :param transpose_weight: [O, I]
        :param keep_k:
        :param asic_input_gloup:
        :return:
        '''
        temp1 = transpose_weight.shape[-1] // asic_input_gloup
        lists = [[] for i in range(asic_input_gloup)]
        for i in range(temp1):
            for j in range(i * asic_input_gloup, (i + 1) * asic_input_gloup):
                lists[j % asic_input_gloup].append(j)
        for i in range(temp1 * asic_input_gloup, transpose_weight.shape[-1]):
            lists[i % asic_input_gloup].append(i)
        temp3 = []
        for i in range(asic_input_gloup):
            value = int(len(lists[i]) / transpose_weight.shape[-1] * keep_k)
            temp3.append(max(value, 1))
        group_mask = np.ones(transpose_weight.shape).transpose([1, 0])
        for i in range(asic_input_gloup):
            temp4 = np.concatenate([transpose_weight[:, one: one + 1] for one in lists[i]], 1)
            mask = _block_sparsity_balance(temp4, temp3[i])
            for one, two in enumerate(lists[i]):
                group_mask[two: two + 1, :] = mask[one: one + 1, :]
        group_mask = group_mask.astype(dtype=transpose_weight.dtype)
        return group_mask

    def find_valid_index(ids, in_size):
        for idx, tensor_idx in enumerate(ids):
            if tensor_idx >= in_size:
                return idx

    def computer_mask(weight, in_size, group_size, block_size, keep_k, patch_size, asic_input_gloup):
        temp1_1 = max(int(np.ceil(in_size / group_size)), 1)
        mask = np.ones(weight.shape, dtype=weight.dtype)
        np_weight = weight
        ids0 = np.arange(block_size)
        ids0 = np.concatenate([ids0 + (row_id * group_size) for row_id in range(temp1_1)])
        keep_k0 = int(block_size * temp1_1 * keep_k / in_size)
        cur_k = keep_k0
        for col_id in range(patch_size):  # cpart
            ids = ids0 + (block_size * col_id)
            temp_v = len(ids)
            if min(ids) >= in_size: break
            if len(ids) > in_size: ids = ids[:in_size]
            ids = ids[:find_valid_index(ids, in_size)] if in_size-1 < max(ids) else ids
            cur_k = int((len(ids) / temp_v) * keep_k0)
            mask[ids, :] = _block_1x1(
                np.transpose(np_weight[ids, :], [1, 0]).astype(dtype=weight.dtype),  # IO --> OI
                cur_k,
                asic_input_gloup
            )
        mask = mask.astype(dtype=weight.dtype)
        return mask

    group_size_max = None
    group_size = None
    block_size = None
    if keep_k >= 1:
        if dtype in ['bf16', 'bfloat16']:
            group_size_max = 256
            group_size = cgb
            block_size = group_size_value // 2
        elif dtype == 'int8':
            group_size_max = 512
            group_size = cgb
            block_size = group_size_value
        in_size, out_size = weight.shape
        if group_size > block_size:
            assert group_size % block_size == 0
        patch_size = max(group_size // block_size, 1)
        if (in_size / patch_size) > group_size_max:
            ori_insize = in_size
            ori_keep_k = keep_k
            inc_group_size = int(np.ceil(in_size / group_size_max))
            temp_list = []
            for i in range(inc_group_size):
                weight_group = weight[i * group_size_max:(i + 1) * group_size_max, :]
                in_size, out_size = weight_group.shape
                keep_k = ori_keep_k * (in_size / ori_insize)
                mask_group = computer_mask(weight_group, in_size, group_size, block_size, keep_k, patch_size,
                                           asic_input_gloup)
                temp_list.append(mask_group)
            mask = np.concatenate(temp_list, 0)
        else:
            if in_size < block_size:
                block_size = in_size
            if group_size < group_size_value and dtype == 'int8':
                group_size = group_size_value
            if group_size < (group_size_value // 2) and dtype == 'bf16':
                group_size = group_size_value // 2

            mask = computer_mask(weight, in_size, group_size, block_size, keep_k, patch_size, asic_input_gloup)
    else:
        mask = np.zeros_like(weight)
    return mask


def prune_dim_2(weight, keep_k, dtype_, cgb, group_size_value=64):
    """
    weight: [O, I]
    keep_k: float
    cgb: int
    dtype: str
    """
    new_params = np.transpose(weight, [1, 0])
    new_shape = new_params.shape
    if new_shape[0] > 2048:
        value = int(np.ceil(new_shape[0] / 2048))
        tmp = []
        for i in range(value):
            new_params_ = new_params[i * 2048:(i + 1) * 2048, :]
            group_keep = keep_k * (new_params_.shape[0] / new_params.shape[0])
            tmp_group_mask = update_mask_asic_2d(weight=new_params_, keep_k=group_keep, dtype=dtype_, cgb=cgb,
                                                 group_size_value=group_size_value)
            tmp.append(tmp_group_mask)
        mask = np.concatenate(tmp, axis=0)
        mask = np.transpose(mask, [1, 0])
    else:
        if new_params.shape[0] == 768 and dtype_ == "int8":
            new_params0 = new_params[:512, :]
            group_keep0 = keep_k * (new_params0.shape[0] / new_params.shape[0]) / 2
            mask0 = update_mask_asic_2d(
                weight=new_params0,
                keep_k=group_keep0,
                dtype=dtype_, cgb=cgb,
                group_size_value=group_size_value
            )
            new_params1 = new_params[512:, :]
            group_keep1 = keep_k * (new_params1.shape[0] / new_params.shape[0]) * 2
            mask1 = update_mask_asic_2d(
                weight=new_params1,
                keep_k=group_keep1,
                dtype=dtype_, cgb=cgb,
                group_size_value=group_size_value
            )
            tmp_mask = np.concatenate([mask0, mask1], 0)
        else:
            tmp_mask = update_mask_asic_2d(weight=new_params, keep_k=keep_k, dtype=dtype_, cgb=cgb,
                                           group_size_value=group_size_value)
        mask = np.transpose(tmp_mask, [1, 0])
    return mask


def prune_dim_3(weight, keep_k_N, cgb, dtype, group_size_value=64):
    """
    weight: [N, K, C]
    keep_k: float
    cgb: int
    dtype: str
    """
    new_mask_list = []
    for idx in range(weight.shape[0]):
        weight_ = weight[idx]
        keep_k = keep_k_N * (1. / weight.shape[0])
        new_params_ = np.transpose(weight_, [1, 0]) # oc ic
        new_shape = new_params_.shape
        if new_shape[0] > 2048:
            value = int(np.ceil(new_shape[0] / 2048))
            tmp = []
            for i in range(value):
                _new_params_ = new_params_[i * 2048:(i + 1) * 2048, :]
                group_keep = keep_k * (_new_params_.shape[0] / new_params_.shape[0])
                tmp_group_mask = update_mask_asic_2d(
                    weight=_new_params_, keep_k=group_keep, dtype=dtype, cgb=cgb, group_size_value=group_size_value
                )
                tmp.append(tmp_group_mask)
            mask = np.concatenate(tmp, axis=0)
            mask = np.transpose(mask, [1, 0])
            mask = np.reshape(mask, [1] + list(mask.shape))
            new_mask_list.append(mask)
        else:
            if new_params_.shape[0] == 768 and dtype == "int8":
                tmp_list = []
                for i in range(2):
                    if i == 0:
                        new_params_tmp = new_params_[:512, :]
                        # group_keep = keep_k * (new_params_tmp.shape[0] / new_params_.shape[0])
                        group_keep = keep_k * (new_params_tmp.shape[0] / new_params_.shape[0]) / 2.
                        tmp_result = update_mask_asic_2d(
                            weight=new_params_tmp, keep_k=group_keep, dtype=dtype, cgb=cgb,
                            group_size_value=group_size_value
                        )
                        tmp_list.append(tmp_result)
                    else:
                        new_params_tmp = new_params_[512:, :]
                        # group_keep = keep_k * (new_params_tmp.shape[0] / new_params_.shape[0])
                        group_keep = keep_k * (new_params_tmp.shape[0] / new_params_.shape[0]) * 2.
                        tmp_result = update_mask_asic_2d(
                            weight=new_params_tmp, keep_k=group_keep, dtype=dtype, cgb=cgb,
                            group_size_value=group_size_value
                        )
                        tmp_list.append(tmp_result)
                tmp_mask = np.concatenate(tmp_list, axis=0)
            else:
                tmp_mask = update_mask_asic_2d(
                    weight=new_params_, keep_k=keep_k, dtype=dtype, cgb=cgb, group_size_value=group_size_value
                )
            mask = np.transpose(tmp_mask, [1, 0])
            mask = np.reshape(mask, [1] + list(mask.shape))
            new_mask_list.append(mask)
    params_mask = np.concatenate(new_mask_list, axis=0)
    return params_mask


def prune_dim_4(weight, keep_k, cgb, dtype, group_size_value=64):
    """
    weight: [H, W, I, O]
    keep_k: float
    cgb: int
    dtype: str
    """
    mask = update_mask_asic_4d(weight=weight, keep_k=keep_k, dtype=dtype, cgb=cgb, group_size_value=group_size_value)
    return mask


def prune_func(weight, sparsity, dtype_info, cgb, group_size_value=64):
    keep_k = max(int(weight.size * (1.0 - sparsity)), 1)
    if len(weight.shape) == 4:
        mask = prune_dim_4(weight=weight, keep_k=keep_k, cgb=cgb, dtype=dtype_info, group_size_value=group_size_value)
    elif len(weight.shape) == 3:
        mask = prune_dim_3(weight=weight, keep_k_N=keep_k, cgb=cgb, dtype=dtype_info, group_size_value=group_size_value)
    elif len(weight.shape) == 2:
        mask = prune_dim_2(weight=weight, keep_k=keep_k, dtype_=dtype_info, cgb=cgb, group_size_value=group_size_value)
    else:
        mask = update_mask(weight=weight, keep_k=keep_k)
    new_params = weight * mask
    return new_params
