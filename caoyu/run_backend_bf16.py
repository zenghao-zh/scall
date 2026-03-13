"""
Run quantized model in bfloat16, with a manual LSTM implementation
that uses basic torch ops (mm, sigmoid, tanh) all supporting bf16.

Usage:
    python /workspace/huada/scall/caoyu/run_backend_bf16.py [--device cuda:0]
"""

import os
import sys
import argparse
import numpy as np
import torch
from torch.nn import Module

# Add project root
pro_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, pro_dir)

import toml
from viterbi_0224 import (
    Model, insert_fakequant_backbone, FakeQuant, match_names, LSTM,
)

# ============================================================
# Paths
# ============================================================
CAOYU_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = "/workspace/huada/task_results/lstm_ctc_crf_optimized_l9_6x_0214/config.toml"
IO_QUANT_PATH = os.path.join(CAOYU_DIR, "layer_9_6x_io_quant_0303.pth")
ACT_SCALES_PATH = os.path.join(CAOYU_DIR, "layer_9_6x_act_scales_0303.pth")
NPZ_PATH = os.path.join(CAOYU_DIR, "placeholder_backend.npz")


# ============================================================
# Manual LSTM — all ops support bf16
# ============================================================

class ManualLSTMRNN(Module):
    """Drop-in replacement for torch.nn.LSTM(input_size, hidden_size, num_layers=1).

    Uses torch.mm / sigmoid / tanh which all natively support bfloat16.
    Parameter names match torch.nn.LSTM exactly so state_dict is compatible.
    """
    def __init__(self, input_size, hidden_size, bias=True, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # Same param names as torch.nn.LSTM
        self.weight_ih_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter('bias_ih_l0', None)
            self.register_parameter('bias_hh_l0', None)

    def forward(self, x, hx=None):
        """
        Args:
            x: (T, N, input_size)
            hx: optional (h_0, c_0), each (1, N, hidden_size)
        Returns:
            output: (T, N, hidden_size)
            (h_n, c_n): each (1, N, hidden_size)
        """
        T, N, _ = x.shape
        H = self.hidden_size

        if hx is not None:
            h = hx[0].squeeze(0)
            c = hx[1].squeeze(0)
        else:
            h = torch.zeros(N, H, dtype=x.dtype, device=x.device)
            c = torch.zeros(N, H, dtype=x.dtype, device=x.device)

        W_ih = self.weight_ih_l0  # (4H, input_size)
        W_hh = self.weight_hh_l0  # (4H, H)
        b = None
        if self.bias:
            b = self.bias_ih_l0 + self.bias_hh_l0  # (4H,)

        outputs = []
        c_states = []
        for t in range(T):
            # gates = x_t @ W_ih^T + h @ W_hh^T + bias
            gates = torch.mm(x[t], W_ih.t()) + torch.mm(h, W_hh.t())
            if b is not None:
                gates = gates + b

            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)

            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)
            c_states.append(c)

        output = torch.stack(outputs, dim=0)  # (T, N, H)
        # Store all cell states for external access (e.g. hooks)
        self._all_c = torch.stack(c_states, dim=0)  # (T, N, H)
        return output, (h.unsqueeze(0), c.unsqueeze(0))


def replace_lstm_with_manual(model):
    """Replace all torch.nn.LSTM inside RNNWrapper.rnn with ManualLSTMRNN,
    copying weights over. Works on the model after insert_fakequant_backbone."""
    replaced = 0
    for name, module in model.named_modules():
        # LSTM layers from viterbi_0224 are RNNWrapper with self.rnn = torch.nn.LSTM
        if isinstance(module, LSTM) and isinstance(module.rnn, torch.nn.LSTM):
            old_rnn = module.rnn
            H = old_rnn.hidden_size
            I = old_rnn.input_size
            manual = ManualLSTMRNN(I, H, bias=old_rnn.bias)
            # Copy parameters
            manual.load_state_dict(old_rnn.state_dict())
            module.rnn = manual
            replaced += 1
    print(f"[replace_lstm_with_manual] Replaced {replaced} torch.nn.LSTM -> ManualLSTMRNN")
    return model


# ============================================================
# Utilities (same as run_backend.py)
# ============================================================

def bf16_to_float32(tensor: np.ndarray):
    tsr = np.left_shift(tensor.view("uint16").astype(np.int32), 16).view(np.float32)
    return tsr


def load_npz_input(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    inp_raw = data["input"]
    inp_u16 = inp_raw[:, :, :4796, :1].copy()
    inp_fp32 = bf16_to_float32(inp_u16)
    inp_tensor = torch.from_numpy(inp_fp32)
    print(f"[Input] shape={inp_tensor.shape}, dtype={inp_tensor.dtype}, "
          f"min={inp_tensor.min().item():.4f}, max={inp_tensor.max().item():.4f}, "
          f"mean={inp_tensor.mean().item():.4f}, std={inp_tensor.std().item():.4f}")
    return inp_tensor


def register_hooks(model):
    """Register one hook per top-level layer (backbone.0, backbone.1, ..., crfencoder)."""
    hook_outputs = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            hook_outputs[name] = out.detach().cpu()
        return hook_fn

    # Hook each direct child of backbone (0~12)
    for name, module in model.backbone._modules.items():
        full_name = f"backbone.{name}"
        hooks.append(module.register_forward_hook(make_hook(full_name)))

    # ---- Detailed hooks for the first LSTM (backbone.4) ----
    # backbone.4 = Sequential(FakeQuant[0], LSTM_wrapper[1], FakeQuant[2])
    #
    # Input FakeQuant (backbone.4.0): layer-to-layer input → int8
    first_lstm_seq = model.backbone._modules["4"]
    input_fq = first_lstm_seq[0]   # FakeQuant (input side)
    output_fq = first_lstm_seq[2]  # FakeQuant (output side)
    lstm_rnn = first_lstm_seq[1].rnn  # ManualLSTMRNN

    def fq_int8_hook(name):
        """Create a hook that saves both dequantized output and int8 values."""
        def hook_fn(module, inp, output):
            hook_outputs[name] = output.detach().cpu()
            scale = module.scale.to(dtype=output.dtype, device=output.device)
            int8_vals = torch.round(output / scale).to(torch.int8)
            hook_outputs[name + "_int8"] = int8_vals.detach().cpu()
        return hook_fn

    hooks.append(input_fq.register_forward_hook(fq_int8_hook("lstm1_input_fq")))
    hooks.append(output_fq.register_forward_hook(fq_int8_hook("lstm1_output_fq")))

    # ManualLSTMRNN: capture h (all timesteps, bf16) and c (all timesteps, bf16)
    def lstm_rnn_hook(module, inp, output):
        # output = (output_seq, (h_n, c_n))
        h_all = output[0].detach().cpu()       # (T, N, H) bf16 — h at every timestep
        hook_outputs["lstm1_h"] = h_all
        # c at every timestep stored by ManualLSTMRNN
        hook_outputs["lstm1_c"] = module._all_c.detach().cpu()  # (T, N, H) bf16
    hooks.append(lstm_rnn.register_forward_hook(lstm_rnn_hook))

    # Hook crfencoder
    hooks.append(model.crfencoder.register_forward_hook(make_hook("crfencoder")))

    return hook_outputs, hooks


# ============================================================
# Load model (same logic as load_quant_model but with manual LSTM swap)
# ============================================================

def load_bf16_quant_model(config, io_quant_path, act_scales_path, device):
    """Load quantized model, replace LSTM with ManualLSTMRNN, convert to bfloat16."""
    device = torch.device(device)
    model = Model(config)

    # Insert FakeQuant layers
    act_scales = torch.load(act_scales_path, map_location=device)
    insert_fakequant_backbone(model, act_scales, bitwidth=8, device=device)

    print(model)

    # Load quantized weights (float32)
    state_dict = torch.load(io_quant_path, map_location=device)
    state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, model).items()}
    model.load_state_dict(state_dict)

    # Replace torch.nn.LSTM -> ManualLSTMRNN (while still float32)
    replace_lstm_with_manual(model)

    # Convert entire model to bfloat16
    model.bfloat16()
    model.eval()
    model.to(device)
    return model


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run quantized model in bf16 with manual LSTM")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    device = args.device

    assert os.path.exists(CONFIG_PATH), f"Config not found: {CONFIG_PATH}"
    config = toml.load(CONFIG_PATH)
    print(f"[Config] loaded from {CONFIG_PATH}")

    # ---- Load model (bf16 with ManualLSTM) ----
    print("[Loading bf16 quantized model with ManualLSTMRNN]")
    model = load_bf16_quant_model(config, IO_QUANT_PATH, ACT_SCALES_PATH, device)
    print(f"[Model] loaded (bfloat16)")
    print(model)

    # ---- Register hooks ----
    hook_outputs, hooks = register_hooks(model)
    print(f"\n[Hooks registered] {len(hooks)} hooks")

    # ---- Load input ----
    print("\n[Loading input from npz]")
    inp = load_npz_input(NPZ_PATH)
    inp = inp.squeeze(-1).bfloat16().to(device)  # (128, 1, 4796) bf16
    print(f"[Input on device] shape={inp.shape}, dtype={inp.dtype}")

    # ---- Forward pass ----
    print("\n[Running forward pass (bfloat16)]")
    with torch.no_grad():
        scores = model(inp)

    # ---- Remove hooks ----
    for h in hooks:
        h.remove()

    # ---- Print captured outputs ----
    print(f"\n[Captured {len(hook_outputs)} layer outputs]")
    print(f"{'Layer':<45} {'Shape':<30} {'dtype':<12} {'min':>10} {'max':>10}")
    print("-" * 110)
    for name, tensor in hook_outputs.items():
        print(f"{name:<45} {str(tuple(tensor.shape)):<30} {str(tensor.dtype):<12} "
              f"{tensor.float().min().item():>10.4f} {tensor.float().max().item():>10.4f}")

    # ---- Save layer outputs as compressed npz ----
    output_path = os.path.join(CAOYU_DIR, "layer_outputs_bf16.npz")
    save_dict = {}
    for name, tensor in hook_outputs.items():
        if tensor.dtype == torch.int8:
            save_dict[name] = tensor.numpy()  # keep int8
        else:
            save_dict[name] = tensor.float().numpy()
    np.savez_compressed(output_path, **save_dict)
    print(f"\n[Saved] {output_path}  ({len(save_dict)} layers)")


if __name__ == "__main__":
    main()
