"""
compiled_lstm.py — 纯 Python 手写 LSTM，配合 torch.compile 速度超越 cuDNN

state_dict 键名与 torch.nn.LSTM 完全一致 (weight_ih_l0, weight_hh_l0, ...)，
可直接替换 nn.LSTM 并加载已有 checkpoint，无需任何键名转换。

用法:
    from compiled_lstm import CompiledLSTM

    # 方式1: 新建
    lstm = CompiledLSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True)
    lstm = lstm.to("cuda")
    lstm = torch.compile(lstm, mode="reduce-overhead")
    output, (h_n, c_n) = lstm(x)

    # 方式2: 从现有 nn.LSTM 转换 (自动复制权重 + 设备)
    fast = CompiledLSTM.from_nn_lstm(existing_nn_lstm)
    fast = torch.compile(fast, mode="reduce-overhead")

    # 方式3: 直接替换模型中的 nn.LSTM 并加载 checkpoint
    model.encoder.rnn = CompiledLSTM(512, 512, 1)  # 替换
    model.load_state_dict(checkpoint)               # 键名完全兼容，直接加载
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple, Optional
import math


class CompiledLSTM(nn.Module):
    """
    纯 Python 多层 LSTM，通过 torch.compile 生成融合 Triton kernel，
    前向速度可达甚至超越 cuDNN。

    与 torch.nn.LSTM 的区别:
        - 不依赖 cuDNN，兼容 torch.compile / torch.vmap
        - state_dict 键名与 nn.LSTM 完全一致，checkpoint 可互通
        - 不支持 bidirectional 和 proj_size（如需要可扩展）

    参数:
        input_size:  输入特征维度
        hidden_size: 隐藏状态维度
        num_layers:  LSTM 层数 (默认 1)
        bias:        是否使用偏置 (默认 True)
        batch_first: 输入是否为 [batch, seq, feature] (默认 False)
        dropout:     层间 dropout 比率 (默认 0, 仅 num_layers > 1 时生效)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout

        # ========================================================
        # 核心: 用与 nn.LSTM 完全一致的参数名注册权重
        # 这样 state_dict 输出的键名自动就是:
        #   weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        #   weight_ih_l1, weight_hh_l1, bias_ih_l1, bias_hh_l1
        #   ...
        # 跟 nn.LSTM 一模一样，checkpoint 直接互通。
        # ========================================================
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.register_parameter(
                f"weight_ih_l{i}",
                nn.Parameter(torch.empty(4 * hidden_size, layer_input_size)),
            )
            self.register_parameter(
                f"weight_hh_l{i}",
                nn.Parameter(torch.empty(4 * hidden_size, hidden_size)),
            )
            if bias:
                self.register_parameter(
                    f"bias_ih_l{i}",
                    nn.Parameter(torch.empty(4 * hidden_size)),
                )
                self.register_parameter(
                    f"bias_hh_l{i}",
                    nn.Parameter(torch.empty(4 * hidden_size)),
                )
            else:
                self.register_parameter(f"bias_ih_l{i}", None)
                self.register_parameter(f"bias_hh_l{i}", None)

        if dropout > 0 and num_layers > 1:
            self.drop = nn.Dropout(dropout)
        else:
            self.drop = None

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        if self.batch_first:
            input = input.transpose(0, 1)  # -> [seq, batch, feature]

        seq_len, batch_size, _ = input.shape

        if hx is None:
            h0 = torch.zeros(
                self.num_layers, batch_size, self.hidden_size,
                dtype=input.dtype, device=input.device,
            )
            c0 = torch.zeros_like(h0)
        else:
            h0, c0 = hx

        h_n_list = []
        c_n_list = []
        cur_input = input

        for layer_idx in range(self.num_layers):
            # 直接用 nn.LSTM 兼容的参数名取权重
            w_ih = getattr(self, f"weight_ih_l{layer_idx}")
            w_hh = getattr(self, f"weight_hh_l{layer_idx}")
            b_ih = getattr(self, f"bias_ih_l{layer_idx}")
            b_hh = getattr(self, f"bias_hh_l{layer_idx}")

            h = h0[layer_idx]
            c = c0[layer_idx]

            # 核心优化: 预计算所有时间步的 input projection
            flat = cur_input.reshape(seq_len * batch_size, cur_input.size(2))
            input_proj = torch.mm(flat, w_ih.t()).reshape(seq_len, batch_size, -1)
            if b_ih is not None:
                input_proj = input_proj + b_ih

            outputs = []
            for t in range(seq_len):
                gates = input_proj[t] + torch.mm(h, w_hh.t())
                if b_hh is not None:
                    gates = gates + b_hh
                i, f, g, o = gates.chunk(4, 1)
                i = torch.sigmoid(i)
                f = torch.sigmoid(f)
                g = torch.tanh(g)
                o = torch.sigmoid(o)
                c = f * c + i * g
                h = o * torch.tanh(c)
                outputs.append(h)

            cur_input = torch.stack(outputs)
            h_n_list.append(h)
            c_n_list.append(c)

            # 层间 dropout（最后一层不加）
            if self.drop is not None and layer_idx < self.num_layers - 1:
                cur_input = self.drop(cur_input)

        output = cur_input
        if self.batch_first:
            output = output.transpose(0, 1)  # -> [batch, seq, feature]

        return output, (torch.stack(h_n_list), torch.stack(c_n_list))

    @staticmethod
    def from_nn_lstm(nn_lstm: nn.LSTM) -> "CompiledLSTM":
        """从 torch.nn.LSTM 实例创建 CompiledLSTM，自动复制所有权重。"""
        ref_param = next(nn_lstm.parameters())
        lstm = CompiledLSTM(
            input_size=nn_lstm.input_size,
            hidden_size=nn_lstm.hidden_size,
            num_layers=nn_lstm.num_layers,
            bias=nn_lstm.bias,
            batch_first=nn_lstm.batch_first,
            dropout=nn_lstm.dropout,
        ).to(device=ref_param.device, dtype=ref_param.dtype)
        # 键名完全一致，直接加载
        lstm.load_state_dict(nn_lstm.state_dict(), strict=False)
        return lstm


# =============================================================================
# 自带 benchmark：python compiled_lstm.py 直接运行
# =============================================================================

def _benchmark(fn, *args, warmup=10, repeats=100, label=""):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5
    print(f"  {label:42s} | {mean:8.3f} ms +/- {std:.3f}")
    return mean


def main():
    if not torch.cuda.is_available():
        print("需要 CUDA GPU!")
        return

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    configs = [
        (512, 512, 1, 100, 64),
        (256, 256, 2, 50, 32),
        (128, 512, 1, 200, 16),
    ]

    for inp, hid, nlayers, seq, batch in configs:
        print(f"\n{'='*72}")
        print(f"  input={inp} hidden={hid} layers={nlayers} seq={seq} batch={batch}")
        print(f"{'='*72}")

        x = torch.randn(seq, batch, inp, device=device)
        h0 = torch.zeros(nlayers, batch, hid, device=device)
        c0 = torch.zeros_like(h0)

        # 基准: nn.LSTM (cuDNN)
        ref = nn.LSTM(inp, hid, nlayers).to(device).eval()

        # 手写 LSTM
        fast = CompiledLSTM.from_nn_lstm(ref).eval()

        # 验证 state_dict 键名一致性
        ref_keys = sorted(ref.state_dict().keys())
        fast_keys = sorted(fast.state_dict().keys())
        assert ref_keys == fast_keys, (
            f"键名不匹配!\n  nn.LSTM: {ref_keys}\n  CompiledLSTM: {fast_keys}"
        )
        print(f"  state_dict 键名验证: OK ({len(ref_keys)} params)")

        # torch.compile 加速
        has_compile = hasattr(torch, "compile")
        if has_compile:
            try:
                fast_compiled = torch.compile(fast, mode="reduce-overhead")
            except Exception as e:
                print(f"  [WARN] torch.compile 不可用: {e}")
                has_compile = False
        if not has_compile:
            fast_compiled = fast

        # 正确性验证
        with torch.no_grad():
            out_ref, _ = ref(x, (h0, c0))
            out_fast, _ = fast(x, (h0, c0))
            diff = (out_ref - out_fast).abs().max().item()
            print(f"  正确性验证: max diff = {diff:.2e} {'OK' if diff < 1e-4 else 'FAIL'}")

        # 速度对比
        print()
        with torch.no_grad():
            t1 = _benchmark(ref, x, (h0, c0), label="nn.LSTM (cuDNN)")
            tag2 = "torch.compile" if has_compile else "pure Python"
            t2 = _benchmark(fast_compiled, x, (h0, c0), warmup=20,
                            label=f"CompiledLSTM ({tag2})")
            t3 = _benchmark(fast, x, (h0, c0), label="CompiledLSTM (no compile)")

        print(f"\n  nn.LSTM (cuDNN)              = {t1:.3f} ms  (baseline)")
        print(f"  CompiledLSTM ({tag2:13s}) = {t2:.3f} ms  ({t2/t1:.2f}x)")
        print(f"  CompiledLSTM (no compile)    = {t3:.3f} ms  ({t3/t1:.2f}x)")


if __name__ == "__main__":
    main()