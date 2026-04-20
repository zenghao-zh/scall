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
moffett_ae.rcp_forward(x)      # 1 / x

# forward + custom backward
moffett_ae.sigmoid(x)
moffett_ae.tanh(x)
moffett_ae.rcp(x)
```

## Notes

- Input must be `torch.bfloat16`.
- CPU and CUDA are both supported.
- Backward uses the standard smooth surrogate gradients:
  - sigmoid: `y * (1 - y)`
  - tanh: `1 - y^2`
  - rcp: `-y^2`  (i.e. `-1 / x^2`)
- `rcp` mirrors the reference `ShiftActiv("1/x")` path:
  - sign-strip the input, normalise `|x|` into `[1.0, 2.0)` via a power-of-two shift,
    evaluate the 4-interval polynomial with 20-bit truncation (`mul20`/`add20`),
    then shift back and re-apply the sign.
  - `1/(+0) = +inf`, `1/(-0) = -inf`, `1/(±inf) = ±0`, `1/NaN = NaN`.
