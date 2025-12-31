import torchinfo

from .cgb_helper import get_node_cgb


def get_max_prune_rate(in_channel):
    if in_channel <= 8:
        return 0.0
    elif in_channel <= 16:
        return 1 / 2
    elif in_channel <= 32:
        return 3 / 4
    elif in_channel <= 64:
        return 7 / 8
    else:
        return 1.0


def generate_prune_dict(model,sparsity, seq_len: int=None):
    prune_dict = {}
    set_up_info = {'cgb': {}, 'dtype': {}}
    # automatic cgb generation, only support CNN
    if seq_len:
        summary_res = torchinfo.summary(model=model,
                                        input_size=(1,1,seq_len),
                                        col_names=[
                                            'kernel_size',
                                            'input_size',
                                            'output_size'
                                            ],
                                        depth=10,
                                        verbose=0)
        layer_infos = []
        for layer_info in summary_res.summary_list:
            if layer_info.class_name in ['Conv1d','Conv2d']:
                info = {
                        'class_name':layer_info.class_name,
                        'kernel_size':layer_info.kernel_size,
                        'input_size':layer_info.input_size,
                        'output_size':layer_info.output_size
                        }
                layer_infos.append(info)
                
    idx = 0
    max_prune_rate = 1.0
    for name, parameter in model.named_parameters():
        # Conv
        if 'weight' in name and len(parameter.shape)>2:
            # # automatic cgb generation, only support CNN
            # if seq_len:
            #     layer_info = layer_infos[idx]
            #     op_type = 'nn.conv2d' if len(parameter.shape) == 4 else 'nn.dense'
            #     in_ch = layer_info['input_size'][1]
            #     out_ch = layer_info['output_size'][1]
            #     in_hw = layer_info['input_size'][-1]*layer_info['input_size'][-2]
            #     out_hw = layer_info['output_size'][-1]*layer_info['output_size'][-2]
            #     cgb_in,cgb_out = get_node_cgb(
            #                                 op_type=op_type,
            #                                 input_channel=in_ch,
            #                                 output_channel=out_ch,
            #                                 data_width=1,
            #                                 input_hw=in_hw,
            #                                 output_hw=out_hw
            #                                 )
            #     set_up_info['cgb'][name] = cgb_out
            #     set_up_info['dtype'][name] = 'int8'
            #     idx += 1
            
            if len(parameter.shape)==3:
                set_up_info['cgb'][name] = 64
                set_up_info['dtype'][name] = 'int8'
            

            if len(parameter.shape)==2:
                if list(parameter.shape)[2:] == [1, 1]:
                    max_prune_rate = get_max_prune_rate(list(parameter.shape)[1])
                else:
                    max_prune_rate = 1.0
            
            prune_dict[name] = min(sparsity, max_prune_rate)
        
        # Linear
        if 'weight' in name and len(parameter.shape)==2:
            if 768 in parameter.shape:
                set_up_info['cgb'][name]=256
                set_up_info['dtype'][name] = 'bf16'
            else:
                set_up_info['cgb'][name] = 512
                set_up_info['dtype'][name] = 'int8'
            max_prune_rate = 1.0
            prune_dict[name] = min(sparsity, max_prune_rate)

    return prune_dict, set_up_info