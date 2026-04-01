# Moffett sigmoid/tanh PyTorch CUDA extension

This package keeps the original piecewise + 20-bit truncation forward path and adds:

- CPU forward
- CUDA forward
- Python-side autograd wrappers for backward

## Install

```bash
pip install -e .
```

or

```bash
python setup.py build_ext --inplace
```

## API

```python
from load_moffett_ae import moffett_ae

# raw forward only
moffett_ae.sigmoid_forward(x)
moffett_ae.tanh_forward(x)

# forward + custom backward
moffett_ae.sigmoid(x)
moffett_ae.tanh(x)
```

## Notes

- Input must be `torch.bfloat16`.
- CPU and CUDA are both supported.
- Backward uses the standard smooth surrogate gradients:
  - sigmoid: `y * (1 - y)`
  - tanh: `1 - y^2`
