from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from .ops import fast_lstm


class FastLSTM(nn.Module):
    """LSTM built from a handwritten forward/backward cell extension."""

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
        proj_size=0,
        activation_impl="moffett",
        device=None,
        dtype=None,
    ):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if proj_size < 0:
            raise ValueError("proj_size must be non-negative")
        if proj_size >= hidden_size and proj_size != 0:
            raise ValueError("proj_size must be smaller than hidden_size")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be between 0 and 1")
        if activation_impl not in {"native", "custom", "formula_fp64", "moffett"}:
            raise ValueError(
                "activation_impl must be 'native', 'custom', 'formula_fp64', or 'moffett'"
            )

        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = float(dropout)
        self.bidirectional = bidirectional
        self.proj_size = proj_size
        self.activation_impl = activation_impl
        self.num_directions = 2 if bidirectional else 1
        self.real_hidden_size = proj_size if proj_size > 0 else hidden_size
        self._cuda_delegate = None

        for layer in range(num_layers):
            layer_input_size = (
                input_size if layer == 0 else self.num_directions * self.real_hidden_size
            )
            for direction in range(self.num_directions):
                suffix = "_reverse" if direction == 1 else ""
                setattr(
                    self,
                    f"weight_ih_l{layer}{suffix}",
                    nn.Parameter(torch.empty(4 * hidden_size, layer_input_size, **factory_kwargs)),
                )
                setattr(
                    self,
                    f"weight_hh_l{layer}{suffix}",
                    nn.Parameter(
                        torch.empty(4 * hidden_size, self.real_hidden_size, **factory_kwargs)
                    ),
                )
                if proj_size > 0:
                    setattr(
                        self,
                        f"weight_hr_l{layer}{suffix}",
                        nn.Parameter(
                            torch.empty(self.real_hidden_size, hidden_size, **factory_kwargs)
                        ),
                    )
                if bias:
                    setattr(
                        self,
                        f"bias_ih_l{layer}{suffix}",
                        nn.Parameter(torch.empty(4 * hidden_size, **factory_kwargs)),
                    )
                    setattr(
                        self,
                        f"bias_hh_l{layer}{suffix}",
                        nn.Parameter(torch.empty(4 * hidden_size, **factory_kwargs)),
                    )
                else:
                    self.register_parameter(f"bias_ih_l{layer}{suffix}", None)
                    self.register_parameter(f"bias_hh_l{layer}{suffix}", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -stdv, stdv)

    def _suffix(self, direction: int) -> str:
        return "_reverse" if direction == 1 else ""

    def _layer_params(self, layer: int, direction: int):
        suffix = self._suffix(direction)
        weight_ih = getattr(self, f"weight_ih_l{layer}{suffix}")
        weight_hh = getattr(self, f"weight_hh_l{layer}{suffix}")
        bias_ih = getattr(self, f"bias_ih_l{layer}{suffix}")
        bias_hh = getattr(self, f"bias_hh_l{layer}{suffix}")
        weight_hr = (
            getattr(self, f"weight_hr_l{layer}{suffix}") if self.proj_size > 0 else None
        )
        return weight_ih, weight_hh, bias_ih, bias_hh, weight_hr

    def _parameter_names(self):
        names = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                suffix = self._suffix(direction)
                names.append(f"weight_ih_l{layer}{suffix}")
                names.append(f"weight_hh_l{layer}{suffix}")
                if self.bias:
                    names.append(f"bias_ih_l{layer}{suffix}")
                    names.append(f"bias_hh_l{layer}{suffix}")
                if self.proj_size > 0:
                    names.append(f"weight_hr_l{layer}{suffix}")
        return names

    def _get_cuda_delegate(self):
        if self.activation_impl != "native":
            return None
        delegate = self.__dict__.get("_cuda_delegate")
        expected_device = self.weight_ih_l0.device
        expected_dtype = self.weight_ih_l0.dtype

        rebuild = delegate is None
        if not rebuild:
            sample = delegate.weight_ih_l0
            rebuild = sample.device != expected_device or sample.dtype != expected_dtype

        if rebuild:
            delegate = nn.LSTM(
                self.input_size,
                self.hidden_size,
                num_layers=self.num_layers,
                bias=self.bias,
                batch_first=self.batch_first,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
                proj_size=self.proj_size,
                device=expected_device,
                dtype=expected_dtype,
            )
            for name in self._parameter_names():
                delegate._parameters[name] = getattr(self, name)
            delegate._update_flat_weights()
            delegate.flatten_parameters()
            object.__setattr__(self, "_cuda_delegate", delegate)

        delegate.training = self.training
        return delegate

    def _activation_mode(self) -> int:
        if self.activation_impl == "native":
            return 0
        if self.activation_impl == "custom":
            return 1
        if self.activation_impl == "formula_fp64":
            return 2
        return 3

    def _init_hidden(self, input):
        batch = input.size(1)
        h0 = input.new_zeros(self.num_layers * self.num_directions, batch, self.real_hidden_size)
        c0 = input.new_zeros(self.num_layers * self.num_directions, batch, self.hidden_size)
        return h0, c0

    def _check_hidden(self, h0, c0, batch, is_batched):
        expected_h = (self.num_layers * self.num_directions, batch, self.real_hidden_size)
        expected_c = (self.num_layers * self.num_directions, batch, self.hidden_size)

        if is_batched:
            if h0.dim() != 3 or c0.dim() != 3:
                raise RuntimeError("For batched input, h0 and c0 must be 3D")
        else:
            if h0.dim() != 2 or c0.dim() != 2:
                raise RuntimeError("For unbatched input, h0 and c0 must be 2D")
            h0 = h0.unsqueeze(1)
            c0 = c0.unsqueeze(1)

        if tuple(h0.shape) != expected_h:
            raise RuntimeError(f"Expected h0 shape {expected_h}, got {tuple(h0.shape)}")
        if tuple(c0.shape) != expected_c:
            raise RuntimeError(f"Expected c0 shape {expected_c}, got {tuple(c0.shape)}")
        return h0, c0

    def forward(self, input, hx=None):
        if isinstance(input, PackedSequence):
            raise NotImplementedError("PackedSequence is not supported yet")
        if input.dim() not in (2, 3):
            raise ValueError(f"FastLSTM expected a 2D or 3D input, got {input.dim()}D")

        module_device = self.weight_ih_l0.device
        if input.device != module_device:
            raise RuntimeError(
                f"Input is on {input.device}, but module parameters are on {module_device}"
            )
        if self.activation_impl == "moffett":
            if input.dtype != torch.bfloat16 or self.weight_ih_l0.dtype != torch.bfloat16:
                raise RuntimeError("activation_impl='moffett' requires torch.bfloat16 inputs and parameters")

        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        if not is_batched:
            input = input.unsqueeze(batch_dim)
        if self.batch_first:
            input = input.transpose(0, 1)

        seq_len, batch, _ = input.shape
        if hx is None:
            h0, c0 = self._init_hidden(input)
        else:
            h0, c0 = self._check_hidden(hx[0], hx[1], batch, is_batched)
            if h0.device != input.device or c0.device != input.device:
                raise RuntimeError("Hidden state device must match the input device")

        if input.is_cuda:
            delegate = self._get_cuda_delegate()
            if delegate is not None:
                delegate_input = input.transpose(0, 1) if self.batch_first else input
                output, hidden = delegate(delegate_input, (h0, c0))
                if not is_batched:
                    output = output.squeeze(batch_dim)
                    hidden = (hidden[0].squeeze(1), hidden[1].squeeze(1))
                return output, hidden

        layer_input = input.contiguous()
        final_h = []
        final_c = []

        for layer in range(self.num_layers):
            layer_outputs = []
            for direction in range(self.num_directions):
                state_index = layer * self.num_directions + direction
                params = self._layer_params(layer, direction)
                dir_output, dir_h, dir_c = fast_lstm(
                    layer_input,
                    h0[state_index].contiguous(),
                    c0[state_index].contiguous(),
                    params[0],
                    params[1],
                    params[2],
                    params[3],
                    params[4],
                    reverse=direction == 1,
                    activation_mode=self._activation_mode(),
                )
                layer_outputs.append(dir_output)
                final_h.append(dir_h)
                final_c.append(dir_c)

            layer_input = (
                layer_outputs[0]
                if self.num_directions == 1
                else torch.cat(layer_outputs, dim=2)
            )
            if self.dropout > 0.0 and self.training and layer < self.num_layers - 1:
                layer_input = F.dropout(layer_input, p=self.dropout, training=True)

        output = layer_input
        h_n = torch.stack(final_h, dim=0)
        c_n = torch.stack(final_c, dim=0)

        if self.batch_first:
            output = output.transpose(0, 1)
        if not is_batched:
            output = output.squeeze(batch_dim)
            h_n = h_n.squeeze(1)
            c_n = c_n.squeeze(1)
        return output, (h_n, c_n)

    @classmethod
    def from_torch_lstm(
        cls, module: nn.LSTM, *, activation_impl: str = "native"
    ) -> "FastLSTM":
        if not isinstance(module, nn.LSTM):
            raise TypeError("module must be an instance of torch.nn.LSTM")

        fast = cls(
            module.input_size,
            module.hidden_size,
            num_layers=module.num_layers,
            bias=module.bias,
            batch_first=module.batch_first,
            dropout=module.dropout,
            bidirectional=module.bidirectional,
            proj_size=module.proj_size,
            activation_impl=activation_impl,
            device=module.weight_ih_l0.device,
            dtype=module.weight_ih_l0.dtype,
        )

        for layer in range(module.num_layers):
            for direction in range(2 if module.bidirectional else 1):
                suffix = "_reverse" if direction == 1 else ""
                getattr(fast, f"weight_ih_l{layer}{suffix}").data.copy_(
                    getattr(module, f"weight_ih_l{layer}{suffix}").data
                )
                getattr(fast, f"weight_hh_l{layer}{suffix}").data.copy_(
                    getattr(module, f"weight_hh_l{layer}{suffix}").data
                )
                if module.bias:
                    getattr(fast, f"bias_ih_l{layer}{suffix}").data.copy_(
                        getattr(module, f"bias_ih_l{layer}{suffix}").data
                    )
                    getattr(fast, f"bias_hh_l{layer}{suffix}").data.copy_(
                        getattr(module, f"bias_hh_l{layer}{suffix}").data
                    )
                if module.proj_size > 0:
                    getattr(fast, f"weight_hr_l{layer}{suffix}").data.copy_(
                        getattr(module, f"weight_hr_l{layer}{suffix}").data
                    )
        return fast
