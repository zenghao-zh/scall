import sys
import os
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


# 在模型中的LSTM层后面插入fake quantization层
def insert_fakequant(model, act_scales, bitwidth,device):
    num_layers = len(model._modules['encoder']._modules)
    print(f"num_layers: {num_layers}")
    for i, (name, module) in enumerate(model._modules['encoder']._modules.items()):
        # replace all linear layers in the model
        if isinstance(module, LSTM):
            if i != num_layers - 2:
                model._modules['encoder']._modules[name] = torch.nn.Sequential(
                    module,
                    FakeQuant(bitwidth, act_scales['encoder.'+ name]["output"].to(device))
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
    config_file = '/workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67/config.toml'
    pretrained_model_file = '/workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67/weights_59.tar'
    act_scales_path = '/workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67/act_scales.pth'
    io_quant_path = '/workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67/io_quant.pth'

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
    data_dir = '/workspace/huada/all_refs_label_for_ctc/train_data/train'
    dataset = TrainingDataSet3(data_dir, tokenization="kmer")
    # 校准只需少量数据，取前2000条
    if len(dataset) > 2000:
        dataset = torch.utils.data.Subset(dataset, range(2000))
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

   

if __name__ == "__main__":
    main()
