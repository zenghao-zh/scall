import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from opencall.models.common.nn import LSTM
from opencall.utils.util import network
import re


class FakeQuant(torch.nn.Module):
    def __init__(self, num_bits, max_val):
        super(FakeQuant, self).__init__()
        self.scale = max_val / (2**(num_bits - 1) - 1)
        self.num_bits = num_bits

    def forward(self, x, return_features=False):
        x = torch.clamp(x / self.scale, -2**(self.num_bits - 1), 2**(self.num_bits - 1) - 1)
        x = torch.round(x)
        x = x * self.scale
        return x


def insert_fakequant(model, act_scales, bitwidth, device):
    """用 act_scales 重建 FakeQuant 结构，使 state_dict key 与 io_quant.pth 对齐"""
    num_layers = len(model._modules['encoder']._modules)
    for i, (name, module) in enumerate(model._modules['encoder']._modules.items()):
        if isinstance(module, LSTM):
            if i != num_layers - 2:
                model._modules['encoder']._modules[name] = torch.nn.Sequential(
                    module,
                    FakeQuant(bitwidth, act_scales['encoder.' + name]["output"].to(device))
                )
    return model


def load_io_quant_model(config_file, io_quant_path, act_scales_path, device="cpu"):
    """
    加载 io_quant.pth 量化模型

    步骤:
      1. 用 config 构建原始模型结构
      2. 用 act_scales 插入 FakeQuant 层，使结构与保存时一致
      3. 加载 io_quant.pth 的 state_dict
    """
    device = torch.device(device)

    # 1. 构建原始模型
    model = network(config_file).to(device)

    # 2. 加载 act_scales 并插入 FakeQuant 层
    act_scales = torch.load(act_scales_path, map_location=device)
    model = insert_fakequant(model, act_scales, bitwidth=8, device=device)

    # 3. 加载量化后的 state_dict
    state_dict = torch.load(io_quant_path, map_location=device)
    model.load_state_dict(state_dict)


    # new_state_dict = {}
    # for k, v in state_dict.items():
    #     # encoder.X.0.rnn.* -> encoder.X.rnn.*
    #     new_key = re.sub(r'(encoder\.\d+)\.0\.(rnn\.)', r'\1.\2', k)
    #     new_state_dict[new_key] = v
    # state_dict = new_state_dict

    model.eval()

    return model, act_scales


if __name__ == "__main__":
    config_file = '/workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67/config.toml'
    io_quant_path = '/workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67/layer_9_6x_io_quant.pth'
    act_scales_path = '/workspace/huada/task_results/lstm_ctc_crf_kmer_0123_67/layer_9_6x_act_scales.pth'
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, act_scales = load_io_quant_model(config_file, io_quant_path, act_scales_path, device)

    print("=== 模型结构 ===")
    print(model)

    print("\n=== act_scales 各层信息 ===")
    for name, scales in act_scales.items():
        input_max = scales["input"].max().item()
        output_max = scales["output"].max().item()
        print(f"  {name}: input_max={input_max:.4f}, output_max={output_max:.4f}")

    print("\n=== state_dict keys ===")
    for k, v in model.state_dict().items():
        print(f"  {k}: {v.shape}")

    print("\n加载成功!")
