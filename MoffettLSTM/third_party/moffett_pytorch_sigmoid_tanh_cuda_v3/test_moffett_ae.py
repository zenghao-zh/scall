import torch
from load_moffett_ae import moffett_ae


def check_forward(name, fn):
    x_cpu = torch.tensor(
        [-45, -44, -15, -1, -0.5, -0.0, 0.0, 0.5, 1, 15, 44, 45],
        dtype=torch.bfloat16,
        device="cpu",
    )
    y_cpu = fn(x_cpu)
    print(f"{name} cpu: ", y_cpu)

    if torch.cuda.is_available():
        x_gpu = x_cpu.cuda()
        y_gpu = fn(x_gpu).cpu()
        print(f"{name} gpu: ", y_gpu)
        print(f"{name} equal: ", torch.equal(y_cpu, y_gpu))
        print(f"{name} max_abs_diff(float): ", (y_cpu.float() - y_gpu.float()).abs().max().item())


def check_backward(name, fn):
    if not torch.cuda.is_available():
        return
    x = torch.randn(4096, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = fn(x)
    loss = y.float().mean()
    loss.backward()
    print(f"{name} backward ok; grad dtype={x.grad.dtype}, grad absmax={x.grad.float().abs().max().item():.6f}")


if __name__ == "__main__":
    check_forward("sigmoid", moffett_ae.sigmoid_forward)
    check_forward("tanh", moffett_ae.tanh_forward)
    check_backward("sigmoid", moffett_ae.sigmoid)
    check_backward("tanh", moffett_ae.tanh)
