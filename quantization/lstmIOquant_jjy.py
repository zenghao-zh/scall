from os.path import join as ospj
import numpy as np 
import torch
from opencall.models.common.nn import LSTM

from opencall.utils.util import (
    network,
    accuracy,
    decode_ref,
    permute,
    load_multi_hd5_v2
)

config = {
    "encoder": {
        "stride": 5,
        "winlen": 19,
        "scale": 5.0,
        "features": 768,
        "activation": "swish",
        "blank_score": 2.0,
        "num_layers": 5
    }
}

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


class DataArgs:
    def __init__(self,limit_train_size,filter_flag=0,tokenization="kmer",shuffle_seed=100,num_wk=16) -> None:
        self.limit_train_size = 1000
        self.filter_flag = 0
        self.tokenization = "kmer"
        self.shuffle_seed = shuffle_seed
        self.num_wk = num_wk


@torch.no_grad()
def main():
    import argparse
    parser = argparse.ArgumentParser(description='LSTM I/O Quantization')
    parser.add_argument('--data_dir', type=str, default='/workspace/huada/scall/train_data',
                        help='Training data directory')
    parser.add_argument('--config_file', type=str, required=True,
                        help='Path to model config.toml file')
    parser.add_argument('--pretrained_model', type=str, required=True,
                        help='Path to pretrained model weights')
    parser.add_argument('--act_scales_path', type=str, default='/workspace/huada/ckpt/act_scales.pth',
                        help='Path to save activation scales')
    parser.add_argument('--io_quant_path', type=str, default='/workspace/huada/ckpt/io_quant.pth',
                        help='Path to save quantized model')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--run_ids', type=str, default='WY_HG002_pc28_hd65_0/ZHANGMENG-20231213163213-0-master',
                        help='Run IDs separated by space')
    parser.add_argument('--data_hours', type=str, default='1_1 1_2',
                        help='Data hours separated by space')
    parser.add_argument('--data_prefix', type=str, default='/workspace/huada/datasets/training_data/',
                        help='Prefix path for datasets')
    args = parser.parse_args()

    # 构建模型
    device = args.device if torch.cuda.is_available() else "cpu"
    model = network(args.config_file).to(device)
    model.load_state_dict(torch.load(args.pretrained_model))
    model.eval()
    orig_model = network(args.config_file).to(device)
    orig_model.load_state_dict(torch.load(args.pretrained_model))
    orig_model.eval()

    # 构建数据集
    print("[loading data]")
    run_ids = args.run_ids.strip().split(" ")
    data_hours = args.data_hours.strip().split("/")
    assert len(run_ids) ==  len(data_hours), "len of run_ids does not equal to len of data_hours"
    sub_datasets = []
    for i in range(len(run_ids)):
        sub_run_hour = data_hours[i].split()
        sub_run_data = [ospj(ospj(args.data_prefix, run_ids[i]), j) for j in sub_run_hour]
        sub_datasets.extend(sub_run_data)
        
    data_args = DataArgs(limit_train_size=10000)
    val_loader = load_multi_hd5_v2(data_args, sub_datasets, args.batch_size, args.data_dir, start_step=0)

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
    
    torch.save(act_scales,args.act_scales_path)
    torch.save(model.state_dict(), args.io_quant_path)

   

if __name__ == "__main__":
    main()

