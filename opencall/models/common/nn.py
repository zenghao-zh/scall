"""
opencall nn modules.
"""

import torch
from torch.nn import Module
from torch.nn.init import orthogonal_
from load_moffett_ae import moffett_ae
from .moffett_lstm import MoffettLSTM
from .complied_lstm import CompiledLSTM


layers = {}


def register(layer):
    layer.name = layer.__name__.lower()
    layers[layer.name] = layer
    return layer


register(torch.nn.ReLU)
register(torch.nn.Tanh)


@register
class Swish(torch.nn.SiLU):
    pass


@register
class Serial(torch.nn.Sequential):
    def __init__(self, sublayers):
        super().__init__(*sublayers)

    def forward(self, x, return_features=False):
        if return_features:
            fmaps = []
            for layer in self:
                x = layer(x)
                fmaps.append(x)
            return x, fmaps
        return super().forward(x)

    def to_dict(self, include_weights=False):
        return {
            "sublayers": [
                to_dict(layer, include_weights) for layer in self._modules.values()
            ]
        }


@register
class Reverse(Module):
    def __init__(self, sublayers):
        super().__init__()
        self.layer = Serial(sublayers) if isinstance(sublayers, list) else sublayers

    def forward(self, x):
        return self.layer(x.flip(0)).flip(0)

    def to_dict(self, include_weights=False):
        if isinstance(self.layer, Serial):
            return self.layer.to_dict(include_weights)
        else:
            return {"sublayers": to_dict(self.layer, include_weights)}


@register
class Convolution(Module):
    def __init__(
        self, insize, size, winlen, stride=1, padding=0, bias=True, activation=None
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
            insize, size, winlen, stride=stride, padding=padding, bias=bias
        )
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.conv.in_channels,
            "size": self.conv.out_channels,
            "bias": self.conv.bias is not None,
            "winlen": self.conv.kernel_size[0],
            "stride": self.conv.stride[0],
            "padding": self.conv.padding[0],
            "activation": self.activation.name if self.activation else None,
        }
        if include_weights:
            res["params"] = {
                "W": self.conv.weight,
                "b": self.conv.bias if self.conv.bias is not None else [],
            }
        return res


@register
class LinearCTCEncoder(Module):
    def __init__(self, insize, outsize=5, bias=True):
        super().__init__()
        self.out_size = outsize
        self.linear = torch.nn.Linear(insize, self.out_size, bias=bias)

    def forward(self, x):
        scores = self.linear(x)
        return scores.log_softmax(2)

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.linear.in_features,
            "outsize": self.out_size,
            "bias": self.linear.bias is not None,
        }
        if include_weights:
            res["params"] = {
                "W": self.linear.weight,
                "b": self.linear.bias if self.linear.bias is not None else [],
            }
        return res


@register
class LinearCRFEncoder(Module):
    def __init__(
        self,
        insize,
        n_base,
        state_len,
        bias=True,
        scale=None,
        activation=None,
        blank_score=None,
        expand_blanks=True,
    ):
        super().__init__()
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = (
            (n_base + 1) * n_base**state_len
            if blank_score is None
            else n_base ** (state_len + 1)
        )
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        scores = self.linear(x)
        if self.activation is not None:
            scores = moffett_ae.tanh(scores.to(torch.bfloat16))
            scores = scores.to(x.dtype)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            T, N, C = scores.shape
            scores = torch.nn.functional.pad(
                scores.view(T, N, C // self.n_base, self.n_base),
                (1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_score,
            ).view(T, N, -1)
        return scores

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.linear.in_features,
            "n_base": self.n_base,
            "state_len": self.state_len,
            "bias": self.linear.bias is not None,
            "scale": self.scale,
            "activation": self.activation.name if self.activation else None,
            "blank_score": self.blank_score,
        }
        if include_weights:
            res["params"] = {
                "W": self.linear.weight,
                "b": self.linear.bias if self.linear.bias is not None else [],
            }
        return res


@register
class LinearCRFEncoder2(Module):
    def __init__(
        self,
        insize,
        n_base,
        state_len,
        bias=True,
        scale=None,
        activation=None,
        blank_score=None,
        expand_blanks=True,
    ):
        super().__init__()
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks
        size = (
            (n_base + 1) * n_base**state_len
            if blank_score is None
            else n_base ** (state_len + 1)
        )
        self.linear_added = torch.nn.Linear(insize, insize, bias=bias)
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()

    def forward(self, x):
        x = self.linear_added(x)
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            T, N, C = scores.shape
            scores = torch.nn.functional.pad(
                scores.view(T, N, C // self.n_base, self.n_base),
                (1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_score,
            ).view(T, N, -1)
        return scores

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.linear.in_features,
            "n_base": self.n_base,
            "state_len": self.state_len,
            "bias": self.linear.bias is not None,
            "scale": self.scale,
            "activation": self.activation.name if self.activation else None,
            "blank_score": self.blank_score,
        }
        if include_weights:
            res["params"] = {
                "W": self.linear.weight,
                "b": self.linear.bias if self.linear.bias is not None else [],
            }
        return res

@register
class Permute(Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
        # return torch.permute(x, self.dims)

    def to_dict(self, include_weights=False):
        return {"dims": self.dims}


def truncated_normal(size, dtype=torch.float32, device=None, num_resample=5):
    x = torch.empty(
        size + (num_resample,), dtype=torch.float32, device=device
    ).normal_()
    i = ((x < 2) & (x > -2)).max(-1, keepdim=True)[1]
    return torch.clamp_(x.gather(-1, i).squeeze(-1), -2, 2)


class RNNWrapper(Module):
    def __init__(
        self,
        rnn_type,
        *args,
        reverse=False,
        orthogonal_weight_init=True,
        disable_state_bias=True,
        bidirectional=False,
        **kwargs,
    ):
        super().__init__()
        if reverse and bidirectional:
            raise Exception(
                "'reverse' and 'bidirectional' should not both be set to True"
            )
        self.reverse = reverse
        self.rnn = rnn_type(*args, **kwargs)
        self.init_orthogonal(orthogonal_weight_init)
        self.init_biases()
        if disable_state_bias:
            self.disable_state_bias()

    def forward(self, x):
        if self.reverse:
            x = x.flip(0)
        y, h = self.rnn(x)
        if self.reverse:
            y = y.flip(0)
        return y

    def init_biases(self, types=("bias_ih",)):
        for name, param in self.rnn.named_parameters():
            if any(k in name for k in types):
                with torch.no_grad():
                    param.set_(
                        0.5
                        * truncated_normal(
                            param.shape, dtype=param.dtype, device=param.device
                        )
                    )

    def init_orthogonal(self, types=True):
        if not types:
            return
        if types == True:
            types = ("weight_ih", "weight_hh")
        for name, x in self.rnn.named_parameters():
            if any(k in name for k in types):
                for i in range(0, x.size(0), self.rnn.hidden_size):
                    orthogonal_(x[i : i + self.rnn.hidden_size])

    def disable_state_bias(self):
        for name, x in self.rnn.named_parameters():
            if "bias_hh" in name:
                x.requires_grad = False
                x.zero_()


class ManualLSTMRNN(Module):
    """Drop-in replacement for torch.nn.LSTM (single-layer, unidirectional).

    Uses torch.mm / sigmoid / tanh which natively support bfloat16 on CUDA,
    bypassing the _thnn_fused_lstm_cell kernel that does not support bf16.
    Parameter names match torch.nn.LSTM so state_dict loads directly.
    
    Memory strategy: Uses gradient checkpointing to trade compute for memory.
    This is essential because manual LSTM cannot match cuDNN's memory efficiency.
    """

    def __init__(self, input_size, hidden_size, bias=True, use_checkpointing=True, 
                 checkpoint_segments=4, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        # Force checkpointing on to save memory
        self.use_checkpointing = use_checkpointing if use_checkpointing is not None else True
        # Divide sequence into N segments for chunked checkpointing
        # Higher = less memory, more recomputation; Lower = more memory, less recomputation
        self.checkpoint_segments = checkpoint_segments
        self.weight_ih_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        if bias:
            self.bias_ih_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size))
            self.bias_hh_l0 = torch.nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter('bias_ih_l0', None)
            self.register_parameter('bias_hh_l0', None)

    @staticmethod
    def _fake_quant_weight(w, num_bits=8):
        """Per-tensor symmetric fake quantization for weight: quantize then dequantize."""
        max_val = w.detach().abs().max()
        qmax = 2 ** (num_bits - 1) - 1
        scale = max_val / qmax
        if scale == 0:
            return w
        w_q = torch.clamp(w / scale, -qmax, qmax)
        w_q = torch.round(w_q)
        return w_q * scale

    @staticmethod
    def _lstm_step(x_t, h, c, w_ih, w_hh, b_ih, b_hh):
        """LSTM cell computation (static for checkpointing)."""
        # Compute gates
        gates = torch.mm(x_t, w_ih.t()) + torch.mm(h, w_hh.t())
        if b_ih is not None:
            gates = gates + b_ih + b_hh
        
        # Split and apply activations
        H = h.size(1)
        i = torch.sigmoid(gates[:, :H])
        f = torch.sigmoid(gates[:, H:2*H])
        g = torch.tanh(gates[:, 2*H:3*H])
        o = torch.sigmoid(gates[:, 3*H:])
        
        # Update states
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new

    def forward(self, x, hx=None):
        T, N, _ = x.shape
        H = self.hidden_size
        
        # Initialize states
        if hx is not None:
            h = hx[0].squeeze(0)
            c = hx[1].squeeze(0)
        else:
            h = torch.zeros(N, H, dtype=x.dtype, device=x.device)
            c = torch.zeros(N, H, dtype=x.dtype, device=x.device)
        
        # Prepare weights
        w_ih = self.weight_ih_l0
        w_hh = self.weight_hh_l0
        b_ih = self.bias_ih_l0 if self.bias else None
        b_hh = self.bias_hh_l0 if self.bias else None
        
        # Pre-allocate output
        outputs = x.new_empty(T, N, H)
        
        if self.use_checkpointing and self.training:
            # Gradient checkpointing: recompute forward during backward
            from torch.utils.checkpoint import checkpoint
            for t in range(T):
                h, c = checkpoint(
                    self._lstm_step,
                    x[t], h, c, w_ih, w_hh, b_ih, b_hh,
                    use_reentrant=False
                )
                outputs[t] = h
        else:
            # Standard forward pass (eval or checkpointing disabled)
            for t in range(T):
                h, c = self._lstm_step(x[t], h, c, w_ih, w_hh, b_ih, b_hh)
                outputs[t] = h
        
        return outputs, (h.unsqueeze(0), c.unsqueeze(0))

@register
class LSTM(RNNWrapper):
    def __init__(self, size, insize, bias=True, reverse=False, dropout=0.0):
        super().__init__(torch.nn.LSTM, size, insize, bias=bias, reverse=reverse, dropout=dropout)

    def to_dict(self, include_weights=False):
        res = {
            "size": self.rnn.hidden_size,
            "insize": self.rnn.input_size,
            "bias": self.rnn.bias,
            "reverse": self.reverse,
        }
        if include_weights:
            res["params"] = {
                "iW": self.rnn.weight_ih_l0.reshape(
                    4, self.rnn.hidden_size, self.rnn.input_size
                ),
                "sW": self.rnn.weight_hh_l0.reshape(
                    4, self.rnn.hidden_size, self.rnn.hidden_size
                ),
                "b": self.rnn.bias_ih_l0.reshape(4, self.rnn.hidden_size),
            }
        return res


def to_dict(layer, include_weights=False):
    if hasattr(layer, "to_dict"):
        return {"type": layer.name, **layer.to_dict(include_weights)}
    return {"type": layer.name}


def from_dict(model_dict, layer_types=None):
    model_dict = model_dict.copy()
    if layer_types is None:
        layer_types = layers
    type_name = model_dict.pop("type")
    typ = layer_types[type_name]
    if "sublayers" in model_dict:
        sublayers = model_dict["sublayers"]
        model_dict["sublayers"] = (
            [from_dict(x, layer_types) for x in sublayers]
            if isinstance(sublayers, list)
            else from_dict(sublayers, layer_types)
        )
    try:
        layer = typ(**model_dict)
    except Exception as e:
        raise Exception(
            f"Failed to build layer of type {typ} with args {model_dict}"
        ) from e
    return layer
