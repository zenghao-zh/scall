from __future__ import annotations

import torch

from ._ext import load_extension


class FastLSTMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, h0, c0, w_ih, w_hh, b_ih, b_hh, w_hr, reverse, activation_mode):
        ext = load_extension()
        (
            output,
            h_n,
            c_n,
            gate_cache,
            cell_cache,
            hidden_raw_cache,
        ) = ext.lstm_forward(
            input, h0, c0, w_ih, w_hh, b_ih, b_hh, w_hr, reverse, activation_mode
        )
        ctx.reverse = reverse
        ctx.activation_mode = activation_mode
        ctx.save_for_backward(
            input,
            output,
            h0,
            c0,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            w_hr,
            gate_cache,
            cell_cache,
            hidden_raw_cache,
        )
        return output, h_n, c_n

    @staticmethod
    def backward(ctx, grad_output, grad_h_n, grad_c_n):
        ext = load_extension()
        (
            input,
            output,
            h0,
            c0,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            w_hr,
            gate_cache,
            cell_cache,
            hidden_raw_cache,
        ) = ctx.saved_tensors

        if grad_h_n is None:
            grad_h_n = torch.zeros_like(h0)
        if grad_c_n is None:
            grad_c_n = torch.zeros_like(c0)

        grads = ext.lstm_backward(
            grad_output,
            grad_h_n,
            grad_c_n,
            input,
            output,
            h0,
            c0,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            w_hr,
            gate_cache,
            cell_cache,
            hidden_raw_cache,
            ctx.reverse,
            ctx.activation_mode,
        )
        (
            grad_input,
            grad_h0,
            grad_c0,
            grad_w_ih,
            grad_w_hh,
            grad_b_ih,
            grad_b_hh,
            grad_w_hr,
        ) = grads
        return (
            grad_input,
            grad_h0,
            grad_c0,
            grad_w_ih,
            grad_w_hh,
            grad_b_ih,
            grad_b_hh,
            grad_w_hr,
            None,
            None,
        )


def fast_lstm(
    input,
    h0,
    c0,
    w_ih,
    w_hh,
    b_ih=None,
    b_hh=None,
    w_hr=None,
    reverse=False,
    activation_mode=0,
):
    if b_ih is None:
        b_ih = input.new_zeros(w_ih.size(0))
    if b_hh is None:
        b_hh = input.new_zeros(w_hh.size(0))
    if w_hr is None:
        w_hr = input.new_empty((0, 0))
    return FastLSTMFunction.apply(
        input, h0, c0, w_ih, w_hh, b_ih, b_hh, w_hr, reverse, activation_mode
    )
