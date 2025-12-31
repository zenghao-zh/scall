def get_default_cgb(channel: int, data_width: int, large_cgb: bool = False, dynamic_input: bool = False,
                    use_smaller_cgb: bool = False, hw: int = None) -> int:
    """

    Args:
        channel: tensor的channel
        data_width: weight的数据类型长度，1: int8，2: bf16
        large_cgb: use cgb > 64 (only in conv2d 1x1 or matmul or dense)
        dynamic_input: ?
        use_smaller_cgb: when cgb >=64, default use smallest cgb
        hw: 输入tensor的h*w

    Returns: 输出cgb

    """
    cgb = int(64 / data_width)
    if channel <= 16 / data_width:
        cgb = int(16 / data_width)
    elif channel <= 32 / data_width:
        cgb = int(32 / data_width)
    else:
        if large_cgb and not dynamic_input:
            if channel <= 64 / data_width:
                cgb = int(64 / data_width)
            if channel <= 128 / data_width:
                cgb = int(128 / data_width)
            if channel <= 256 / data_width:
                cgb = int(256 / data_width)
            if channel > 256 / data_width:
                if channel % (int(512 / data_width)) == 0:
                    cgb = int(512 / data_width)
                if channel % (int(256 / data_width)) == 0:
                    cgb = int(256 / data_width)
                if channel % (int(128 / data_width)) == 0:
                    cgb = int(128 / data_width)
            if use_smaller_cgb and hw:  # HW is HxW
                if hw % 64 == 0 and cgb >= int(64 / data_width):
                    cgb = int(64 / data_width)
                elif hw % 32 == 0 and cgb >= int(128 / data_width):
                    cgb = int(128 / data_width)
                elif hw % 16 == 0 and cgb >= int(256 / data_width):
                    cgb = int(256 / data_width)
                elif hw % 8 == 0 and cgb >= int(512 / data_width):
                    cgb = int(512 / data_width)
        else:
            cgb = int(64 / data_width)
    return cgb


def use_large_cgb(op_type) -> bool:
    """

    Args:
        op_type: the op_type of the node

    Returns: 是否使用大cgb

    """
    if op_type == "nn.dense":
        return True
    else:
        return False


def get_node_cgb(op_type: str, input_channel: int, output_channel: int, data_width: int, input_hw: int,
                 output_hw: int) -> tuple:
    """
    计算 cgb_in 和 cgb_out

    Args:
        op_type: operator type, such as nn.conv2d, nn.dense, nn.batch_matmul
        input_channel: input channel of the node
        output_channel: output channel of the node
        data_width: 1 if int8, 2 if bfloat16
        input_hw: h * w of the input tensor
        output_hw: h * w of the output tensor

    Returns:
        tuple: cgb_in and cgb_out

    """
    large_cgb = use_large_cgb(op_type)
    cgb_in = get_default_cgb(input_channel, data_width, large_cgb=large_cgb, hw=input_hw)
    cgb_out = get_default_cgb(output_channel, data_width, large_cgb=large_cgb, hw=output_hw)
    if op_type == "nn.dense":
        cgb_in = int(512 / data_width)
    if cgb_in > 64 / data_width:
        cgb_out = cgb_in
    return cgb_in, cgb_out
