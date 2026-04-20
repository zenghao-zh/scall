import os
import importlib
import torch
from torch.utils.cpp_extension import CUDA_HOME, load

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXT_NAME = "moffett_ae_sigmoid_tanh_cuda"


def _load_ext():
    try:
        return importlib.import_module(_EXT_NAME)
    except ImportError:
        pass

    if CUDA_HOME is None:
        raise RuntimeError(
            "CUDA toolkit not found. Install CUDA and make sure CUDA_HOME is visible before building this extension."
        )

    return load(
        name=_EXT_NAME,
        sources=[
            os.path.join(_THIS_DIR, "moffett_ae_torch.cpp"),
            os.path.join(_THIS_DIR, "ae_sigmoid_tanh_rcp_cpu.cpp"),
            os.path.join(_THIS_DIR, "ae_sigmoid_tanh_rcp_cuda.cu"),
        ],
        extra_include_paths=[_THIS_DIR],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
        verbose=True,
    )


_ext = _load_ext()


class _MoffettSigmoidFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.dtype != torch.bfloat16:
            raise TypeError("moffett_ae.sigmoid expects torch.bfloat16 input")
        y = _ext.sigmoid_forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        grad = grad_output.float() * y.float() * (1.0 - y.float())
        return grad.to(grad_output.dtype)


class _MoffettTanhFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.dtype != torch.bfloat16:
            raise TypeError("moffett_ae.tanh expects torch.bfloat16 input")
        y = _ext.tanh_forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        grad = grad_output.float() * (1.0 - y.float() * y.float())
        return grad.to(grad_output.dtype)


class _MoffettRcpFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if x.dtype != torch.bfloat16:
            raise TypeError("moffett_ae.rcp expects torch.bfloat16 input")
        y = _ext.rcp_forward(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        (y,) = ctx.saved_tensors
        # d(1/x)/dx = -1/x^2 = -y^2
        grad = -grad_output.float() * y.float() * y.float()
        return grad.to(grad_output.dtype)


class _MoffettAE:
    def sigmoid_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ext.sigmoid_forward(x)

    def tanh_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ext.tanh_forward(x)

    def rcp_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ext.rcp_forward(x)

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return _MoffettSigmoidFn.apply(x)

    def tanh(self, x: torch.Tensor) -> torch.Tensor:
        return _MoffettTanhFn.apply(x)

    def rcp(self, x: torch.Tensor) -> torch.Tensor:
        return _MoffettRcpFn.apply(x)


moffett_ae = _MoffettAE()
