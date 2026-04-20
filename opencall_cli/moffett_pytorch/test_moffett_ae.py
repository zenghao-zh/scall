import torch
from load_moffett_ae import moffett_ae


TORCH_REF = {
    "sigmoid": torch.sigmoid,
    "tanh": torch.tanh,
    "rcp": torch.reciprocal,
}

# Forward inputs per op. rcp avoids exact zero so we don't compare against +/-inf.
INPUTS = {
    "sigmoid": [-45, -44, -15, -1, -0.5, -0.0, 0.0, 0.5, 1, 15, 44, 45],
    "tanh":    [-45, -44, -15, -1, -0.5, -0.0, 0.0, 0.5, 1, 15, 44, 45],
    "rcp":     [-45, -15, -2.0, -1.5, -1.0, -0.5, -0.125, 0.125, 0.5, 1.0, 1.5, 2.0, 15, 45],
}


def _diff_stats(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


def check_forward(name, fn):
    x_cpu = torch.tensor(
        INPUTS[name],
        dtype=torch.bfloat16,
        device="cpu",
    )
    y_cpu = fn(x_cpu)
    y_torch_cpu = TORCH_REF[name](x_cpu)
    print(f"{name} cpu moffett: ", y_cpu)
    print(f"{name} cpu torch  : ", y_torch_cpu)
    max_d, mean_d = _diff_stats(y_cpu, y_torch_cpu)
    print(f"{name} cpu vs torch  max_abs_diff={max_d:.6f}  mean_abs_diff={mean_d:.6f}")

    if torch.cuda.is_available():
        x_gpu = x_cpu.cuda()
        y_gpu = fn(x_gpu).cpu()
        y_torch_gpu = TORCH_REF[name](x_gpu).cpu()
        print(f"{name} gpu moffett: ", y_gpu)
        print(f"{name} gpu torch  : ", y_torch_gpu)
        print(f"{name} cpu/gpu equal (moffett): ", torch.equal(y_cpu, y_gpu))
        print(f"{name} cpu/gpu max_abs_diff(float): ", (y_cpu.float() - y_gpu.float()).abs().max().item())
        max_d, mean_d = _diff_stats(y_gpu, y_torch_gpu)
        print(f"{name} gpu vs torch  max_abs_diff={max_d:.6f}  mean_abs_diff={mean_d:.6f}")


def check_backward(name, fn):
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    x_data = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    if name == "rcp":
        # Push values away from 0 so 1/x and its gradient stay finite.
        sign = torch.where(x_data >= 0, torch.ones_like(x_data), -torch.ones_like(x_data))
        x_data = sign * (x_data.abs() + 0.5)

    x_moffett = x_data.clone().detach().requires_grad_(True)
    y_moffett = fn(x_moffett)
    y_moffett.float().mean().backward()

    x_torch = x_data.clone().detach().requires_grad_(True)
    y_torch = TORCH_REF[name](x_torch)
    y_torch.float().mean().backward()

    fwd_max, fwd_mean = _diff_stats(y_moffett.detach(), y_torch.detach())
    grad_max, grad_mean = _diff_stats(x_moffett.grad, x_torch.grad)
    print(
        f"{name} backward ok; grad dtype={x_moffett.grad.dtype}, "
        f"grad absmax={x_moffett.grad.float().abs().max().item():.6f}"
    )
    print(
        f"{name} fwd  vs torch  max_abs_diff={fwd_max:.6f}  mean_abs_diff={fwd_mean:.6f}"
    )
    print(
        f"{name} grad vs torch  max_abs_diff={grad_max:.6f}  mean_abs_diff={grad_mean:.6f}"
    )


if __name__ == "__main__":
    check_forward("sigmoid", moffett_ae.sigmoid_forward)
    check_forward("tanh", moffett_ae.tanh_forward)
    check_forward("rcp", moffett_ae.rcp_forward)
    check_backward("sigmoid", moffett_ae.sigmoid)
    check_backward("tanh", moffett_ae.tanh)
    check_backward("rcp", moffett_ae.rcp)
