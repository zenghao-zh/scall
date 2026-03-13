import sys
import os
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np 
import torch
from opencall.models.common.nn import LSTM
from opencall.data_loader.data import TrainingDataSet3

from opencall.utils.util import (
    network,
    accuracy,
    decode_ref,
    permute,
)


def validate_one_step(batch, model, device, min_coverage=0.5):
    data, targets, lengths = batch 
    scores = model(data.to(device))
    model = model
    if hasattr(model, 'decode_batch'):
        seqs = model.decode_batch(scores)
    else:
        seqs = [model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
    refs = [decode_ref(target, model.alphabet) for target in targets]
    accs = [
        accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
    ]
    return seqs, refs, accs 


def validate_one_epoch(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        seqs, refs, accs = zip(*(validate_one_step(batch,model,device) for batch in dataloader))
    seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
    return  np.mean(accs), np.median(accs)


# 定义fake quantization层
class FakeQuant(torch.nn.Module):
    def __init__(self, num_bits, max_val):
        super(FakeQuant, self).__init__()
        self.scale = max_val / (2**(num_bits - 1) - 1)
        self.num_bits = num_bits

    def forward(self, x,return_features=False):
        # quantize the input tensor x to the bitwidth
        x = torch.clamp(x / self.scale, -2**(self.num_bits - 1), 2**(self.num_bits - 1) - 1)
        x = torch.round(x)
        # dequantize the tensor x
        x = x * self.scale
        return x


# 在模型中的所有LSTM层插入fake quantization层，确保每层输入输出都被量化
def insert_fakequant(model, act_scales, bitwidth, device):
    num_layers = len(model._modules['encoder']._modules)
    print(f"num_layers: {num_layers}")
    # 找出所有LSTM层的索引
    lstm_entries = [
        (i, name, module) for i, (name, module)
        in enumerate(model._modules['encoder']._modules.items())
        if isinstance(module, LSTM)
    ]
    for i, (name, module) in enumerate(model._modules['encoder']._modules.items()):
        if isinstance(module, LSTM):
            scale_key = 'encoder.' + name
            model._modules['encoder']._modules[name] = torch.nn.Sequential(
                module,
                FakeQuant(bitwidth, act_scales[scale_key]["output"].to(device))
            )
    return model


def hook_model(model, act_scales):
    def stat_hook(name, act_scales):
        def stat_func(self, x, y):
            if isinstance(x, tuple):
                x = x[0]
            if isinstance(y, tuple):
                y = y[0]
            hidden_dim = x.shape[-1]
            x = x.contiguous().view(-1, hidden_dim).abs().detach()
            comming_max = torch.max(x, dim=0)[0].float().cpu()
            y = y.contiguous().view(-1, y.shape[-1]).abs().detach()
            comming_max_y = torch.max(y, dim=0)[0].float().cpu()
            
            if name in act_scales:
                act_scales[name]["input"] = torch.max(act_scales[name]["input"], comming_max)
                act_scales[name]["output"] = torch.max(act_scales[name]["output"], comming_max_y)
            else:
                act_scales[name] = {"input": comming_max, "output": comming_max_y}
        return stat_func

    hooks = list()
    for name, module in model.named_modules():
        if isinstance(module, (LSTM, torch.nn.Linear)):
            hooks.append(module.register_forward_hook(stat_hook(name, act_scales=act_scales)))
    return hooks



@torch.no_grad()
def main():
    config_file = '/workspace/huada/task_results/lstm_ctc_crf_qat_int8/config.toml'
    pretrained_model_file = '/workspace/huada/task_results/lstm_ctc_crf_qat_int8/weights_8.tar'
    act_scales_path = '/workspace/huada/task_results/lstm_ctc_crf_qat_int8/act_scales_8.pth'
    io_quant_path = '/workspace/huada/task_results/lstm_ctc_crf_qat_int8/io_quant_8.pth'
    new_io_quant_path = '/workspace/huada/task_results/lstm_ctc_crf_qat_int8/io_quant_wo0_8.pth'

    # config_file = '/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/config.toml'
    # pretrained_model_file = '/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/weights_40.tar'
    # act_scales_path = '/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/layer_9_6x_act_scales_40.
    # pth'
    # io_quant_path = '/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/layer_9_6x_io_quant_40.pth'
    # new_io_quant_path = '/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/
    # layer_9_6x_io_quant_wo0_40.pth'

    # 构建模型
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = network(config_file).to(device)
    model.load_state_dict(torch.load(pretrained_model_file))
    model.eval()
    orig_model = network(config_file).to(device)
    orig_model.load_state_dict(torch.load(pretrained_model_file))
    orig_model.eval()

    # 构建数据集
    print("[loading data]")
    batch_size = 32
    data_dir = '/workspace/huada/moffett_data/250F600274011_train_data/train'
    dataset = TrainingDataSet3(data_dir, tokenization="kmer")
    # 校准只需少量数据，取前2000条
    if len(dataset) > 20000:
        indices = torch.randperm(len(dataset))[:20000].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)
    val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4, pin_memory=True, shuffle=False)

    act_scales = dict()
    hooks = hook_model(model, act_scales)
    for batch in val_loader:
        batch = batch[0].to(device)
        model(batch)

    for hook in hooks:
        hook.remove()   

    model = insert_fakequant(model, act_scales, 8, device)
    print(f" quant model:\n{model}")

    #验证模型精度
    mean_acc, medium_acc = validate_one_epoch(model, val_loader,device)
    orig_mean_acc, orig_medium_acc = validate_one_epoch(orig_model, val_loader,device)
    print(f"quantized model mean_acc:{mean_acc}, medium_acc:{medium_acc}")
    print(f"original model mean_acc:{orig_mean_acc}, medium_acc:{orig_medium_acc}")
    
    torch.save(act_scales,act_scales_path)
    torch.save(model.state_dict(), io_quant_path)


    state_dict = torch.load(io_quant_path, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        # encoder.X.{0或1}.rnn.* -> encoder.X.rnn.*（第一个LSTM的rnn在索引1，其余在索引0）
        new_key = re.sub(r'(encoder\.\d+)\.\d+\.(rnn\.)', r'\1.\2', k)
        new_state_dict[new_key] = v
    torch.save(new_state_dict, new_io_quant_path)

    


    ## remove .0
    

   

if __name__ == "__main__":
    main()
