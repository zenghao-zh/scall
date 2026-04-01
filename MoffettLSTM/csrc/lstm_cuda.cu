#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <vector>

#include "util_common.hpp"

namespace {

constexpr int64_t kActivationNative = 0;
constexpr int64_t kActivationCustom = 1;
constexpr int64_t kActivationFormulaFp64 = 2;
constexpr int64_t kActivationMoffett = 3;

__device__ __constant__ moffett::U32 kMoffettSigmoidHexDev
    [moffett::kSigmoidRows * moffett::kTableCols] = {
        0xff800000,0xc0a00000,0x00000000,0x00000000,0x00000000,
        0xc0a00000,0xc0900000,0x3b824000,0x3d3da000,0x3e0e2000,
        0xc0900000,0xc0600000,0x3c0aa000,0x3dafe000,0x3e679000,
        0xc0600000,0xc0200000,0x3ca6b000,0x3e2c6000,0x3ebd2000,
        0xc0200000,0xbfc00000,0x3d206000,0x3e868000,0x3ef9f000,
        0xbfc00000,0xbdcd0000,0x3d1f5000,0x3e8b4000,0x3f00b000,
        0xbdcd0000,0x80000000,0x3b908000,0x3e802000,0x3f003000,
        0x00000000,0x3dcc0000,0xbb290000,0x3e800000,0x3f008000,
        0x3dcc0000,0x40000000,0xbd281000,0x3e8c4000,0x3efed000,
        0x40000000,0x40400000,0xbcd2f000,0x3e4c3000,0x3f16b000,
        0x40400000,0x40a00000,0xbbf63000,0x3da23000,0x3f490000,
        0x40a00000,0x7f800000,0x00000000,0x00000000,0x3f800000,
};

__device__ __constant__ moffett::U32 kMoffettTanhHexDev
    [moffett::kTanhRows * moffett::kTableCols] = {
        0x00000000,0x00800000,0x00000000,0x00000000,0x00000000,
        0x00800000,0x32000000,0x00000000,0x3f800000,0x00000000,
        0x32000000,0x3e000000,0x325f3000,0x3f800000,0x18906000,
        0x3e000000,0x3f000000,0xbe89a000,0x3f890000,0xbbb69000,
        0x3f000000,0x3fc00000,0xbe9ef000,0x3f87d000,0x3c586000,
        0x3fc00000,0x40200000,0xbd94e000,0x3ebe2000,0x3f038000,
        0x40200000,0x40500000,0xbc6f7000,0x3dc85000,0x3f558000,
        0x40500000,0x7f800000,0x00000000,0x00000000,0x3f800000,
};

bool has_projection(const torch::Tensor& weight_hr) {
  return weight_hr.defined() && weight_hr.numel() > 0;
}

void check_activation_mode(int64_t activation_mode) {
  TORCH_CHECK(
      activation_mode == kActivationNative ||
          activation_mode == kActivationCustom ||
          activation_mode == kActivationFormulaFp64 ||
          activation_mode == kActivationMoffett,
      "Unsupported activation mode: ",
      activation_mode);
}

int64_t time_index(int64_t step, int64_t seq_len, bool reverse) {
  return reverse ? (seq_len - 1 - step) : step;
}

torch::Tensor reconstruct_prev_hidden(
    const torch::Tensor& output,
    const torch::Tensor& h0,
    bool reverse) {
  auto seq_len = output.size(0);
  auto prev_hidden = torch::empty_like(output);
  if (reverse) {
    if (seq_len > 1) {
      prev_hidden.slice(0, 0, seq_len - 1)
          .copy_(output.slice(0, 1, seq_len));
    }
    prev_hidden.select(0, seq_len - 1).copy_(h0);
  } else {
    prev_hidden.select(0, 0).copy_(h0);
    if (seq_len > 1) {
      prev_hidden.slice(0, 1, seq_len)
          .copy_(output.slice(0, 0, seq_len - 1));
    }
  }
  return prev_hidden;
}

template <typename scalar_t>
__device__ inline scalar_t sigmoid_device(
    scalar_t x,
    int64_t activation_mode) {
  if (activation_mode == kActivationMoffett) {
    auto y = moffett::sigmoid_one_from_table(
        moffett::BF16(static_cast<float>(x)), kMoffettSigmoidHexDev);
    return static_cast<scalar_t>(static_cast<float>(y));
  }
  if (activation_mode == kActivationFormulaFp64) {
    double xd = static_cast<double>(x);
    double y = xd >= 0.0 ? 1.0 / (1.0 + exp(-xd))
                         : exp(xd) / (1.0 + exp(xd));
    return static_cast<scalar_t>(y);
  }
  return static_cast<scalar_t>(1) /
      (static_cast<scalar_t>(1) + exp(-x));
}

template <typename scalar_t>
__device__ inline scalar_t tanh_device(
    scalar_t x,
    int64_t activation_mode) {
  if (activation_mode == kActivationMoffett) {
    auto y = moffett::tanh_one_from_table(
        moffett::BF16(static_cast<float>(x)), kMoffettTanhHexDev);
    return static_cast<scalar_t>(static_cast<float>(y));
  }
  if (activation_mode == kActivationFormulaFp64) {
    double xd = static_cast<double>(x);
    double z = exp(-2.0 * fabs(xd));
    double y = copysign((1.0 - z) / (1.0 + z), xd);
    return static_cast<scalar_t>(y);
  }
  return tanh(x);
}

template <typename scalar_t>
__global__ void lstm_forward_pointwise_kernel(
    const scalar_t* __restrict__ gates,
    const scalar_t* __restrict__ c_prev,
    scalar_t* __restrict__ gate_cache,
    scalar_t* __restrict__ cell_cache,
    scalar_t* __restrict__ hidden_raw,
    scalar_t* __restrict__ output,
    int64_t batch,
    int64_t hidden,
    int64_t activation_mode) {
  auto idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = batch * hidden;
  if (idx >= total) {
    return;
  }

  auto b = idx / hidden;
  auto h = idx % hidden;
  auto gate_row = b * 4 * hidden;
  auto cell_offset = b * hidden + h;

  auto i = sigmoid_device(gates[gate_row + h], activation_mode);
  auto f = sigmoid_device(gates[gate_row + hidden + h], activation_mode);
  auto g = tanh_device(gates[gate_row + 2 * hidden + h], activation_mode);
  auto o = sigmoid_device(gates[gate_row + 3 * hidden + h], activation_mode);

  auto c = f * c_prev[cell_offset] + i * g;
  auto h_raw = o * tanh_device(c, activation_mode);

  gate_cache[gate_row + h] = i;
  gate_cache[gate_row + hidden + h] = f;
  gate_cache[gate_row + 2 * hidden + h] = g;
  gate_cache[gate_row + 3 * hidden + h] = o;
  cell_cache[cell_offset] = c;
  if (hidden_raw != nullptr) {
    hidden_raw[cell_offset] = h_raw;
  }
  if (output != nullptr) {
    output[cell_offset] = h_raw;
  }
}

template <typename scalar_t>
__global__ void lstm_backward_pointwise_kernel(
    const scalar_t* __restrict__ gate_cache,
    const scalar_t* __restrict__ cell_cache,
    const scalar_t* __restrict__ c_prev,
    const scalar_t* __restrict__ grad_h_raw,
    const scalar_t* __restrict__ grad_c_in,
    scalar_t* __restrict__ grad_gates,
    scalar_t* __restrict__ grad_c_out,
    int64_t batch,
    int64_t hidden,
    int64_t activation_mode) {
  auto idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto total = batch * hidden;
  if (idx >= total) {
    return;
  }

  auto b = idx / hidden;
  auto h = idx % hidden;
  auto gate_row = b * 4 * hidden;
  auto cell_offset = b * hidden + h;

  auto i = gate_cache[gate_row + h];
  auto f = gate_cache[gate_row + hidden + h];
  auto g = gate_cache[gate_row + 2 * hidden + h];
  auto o = gate_cache[gate_row + 3 * hidden + h];
  auto c = cell_cache[cell_offset];
  auto c_prev_val = c_prev[cell_offset];
  auto dh = grad_h_raw[cell_offset];
  auto tanh_c = tanh_device(c, activation_mode);
  auto dc = grad_c_in[cell_offset] + dh * o * (1 - tanh_c * tanh_c);

  auto d_o_pre = (dh * tanh_c) * o * (1 - o);
  auto d_i_pre = (dc * g) * i * (1 - i);
  auto d_f_pre = (dc * c_prev_val) * f * (1 - f);
  auto d_g_pre = (dc * i) * (1 - g * g);

  grad_gates[gate_row + h] = d_i_pre;
  grad_gates[gate_row + hidden + h] = d_f_pre;
  grad_gates[gate_row + 2 * hidden + h] = d_g_pre;
  grad_gates[gate_row + 3 * hidden + h] = d_o_pre;
  grad_c_out[cell_offset] = dc * f;
}

void launch_forward_pointwise(
    const torch::Tensor& gates,
    const torch::Tensor& c_prev,
    const torch::Tensor& gate_cache,
    const torch::Tensor& cell_cache,
    const torch::Tensor& hidden_raw,
    const torch::Tensor& output,
    int64_t activation_mode) {
  const auto batch = gates.size(0);
  const auto hidden = c_prev.size(1);
  const int threads = 256;
  const int blocks = static_cast<int>((batch * hidden + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gates.scalar_type(), "lstm_forward_pointwise_cuda", [&] {
        auto hidden_raw_ptr =
            hidden_raw.defined() && hidden_raw.numel() > 0
            ? hidden_raw.data_ptr<scalar_t>()
            : nullptr;
        auto output_ptr = output.defined() && output.numel() > 0
            ? output.data_ptr<scalar_t>()
            : nullptr;
        lstm_forward_pointwise_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            gates.data_ptr<scalar_t>(),
            c_prev.data_ptr<scalar_t>(),
            gate_cache.data_ptr<scalar_t>(),
            cell_cache.data_ptr<scalar_t>(),
            hidden_raw_ptr,
            output_ptr,
            batch,
            hidden,
            activation_mode);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_backward_pointwise(
    const torch::Tensor& gate_cache,
    const torch::Tensor& cell_cache,
    const torch::Tensor& c_prev,
    const torch::Tensor& grad_h_raw,
    const torch::Tensor& grad_c_next,
    const torch::Tensor& grad_gates,
    const torch::Tensor& grad_c_out,
    int64_t activation_mode) {
  const auto batch = cell_cache.size(0);
  const auto hidden = cell_cache.size(1);
  const int threads = 256;
  const int blocks = static_cast<int>((batch * hidden + threads - 1) / threads);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      gate_cache.scalar_type(), "lstm_backward_pointwise_cuda", [&] {
        lstm_backward_pointwise_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            gate_cache.data_ptr<scalar_t>(),
            cell_cache.data_ptr<scalar_t>(),
            c_prev.data_ptr<scalar_t>(),
            grad_h_raw.data_ptr<scalar_t>(),
            grad_c_next.data_ptr<scalar_t>(),
            grad_gates.data_ptr<scalar_t>(),
            grad_c_out.data_ptr<scalar_t>(),
            batch,
            hidden,
            activation_mode);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

std::vector<torch::Tensor> lstm_forward_cuda(
    const torch::Tensor& input,
    const torch::Tensor& h0,
    const torch::Tensor& c0,
    const torch::Tensor& w_ih,
    const torch::Tensor& w_hh,
    const torch::Tensor& b_ih,
    const torch::Tensor& b_hh,
    const torch::Tensor& w_hr,
    bool reverse,
    int64_t activation_mode) {
  check_activation_mode(activation_mode);
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(
      h0.is_cuda() && c0.is_cuda() && w_ih.is_cuda() && w_hh.is_cuda() &&
          b_ih.is_cuda() && b_hh.is_cuda(),
      "all tensors must be CUDA tensors");
  if (activation_mode == kActivationMoffett) {
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16,
        "moffett activation expects torch.bfloat16 input");
  }

  c10::cuda::CUDAGuard device_guard(input.device());

  auto input_c = input.contiguous();
  auto h_prev = h0.contiguous();
  auto c_prev = c0.contiguous();
  auto w_ih_c = w_ih.contiguous();
  auto w_hh_c = w_hh.contiguous();
  auto w_ih_t = w_ih_c.t().contiguous();
  auto w_hh_t = w_hh_c.t().contiguous();
  const auto use_proj = has_projection(w_hr);
  auto w_hr_c = use_proj ? w_hr.contiguous() : torch::Tensor();
  auto w_hr_t = use_proj ? w_hr_c.t().contiguous() : torch::Tensor();
  auto b_total = (b_ih + b_hh).contiguous();

  const auto seq_len = input_c.size(0);
  const auto batch = input_c.size(1);
  const auto input_size = input_c.size(2);
  const auto hidden = c_prev.size(1);
  const auto real_hidden = h_prev.size(1);

  auto options = input_c.options();
  auto output = torch::empty({seq_len, batch, real_hidden}, options);
  auto gate_cache = torch::empty({seq_len, batch, 4 * hidden}, options);
  auto cell_cache = torch::empty({seq_len, batch, hidden}, options);
  auto hidden_raw_cache = use_proj
      ? torch::empty({seq_len, batch, hidden}, options)
      : torch::empty({0, 0, 0}, options);

  auto input_projection =
      torch::mm(input_c.reshape({seq_len * batch, input_size}), w_ih_t)
          .reshape({seq_len, batch, 4 * hidden});
  input_projection = input_projection + b_total.view({1, 1, 4 * hidden});

  for (int64_t step = 0; step < seq_len; ++step) {
    const auto t = time_index(step, seq_len, reverse);
    auto gates = input_projection[t] + torch::mm(h_prev, w_hh_t);
    auto gate_t = gate_cache[t];
    auto cell_t = cell_cache[t];

    if (use_proj) {
      auto h_raw_t = hidden_raw_cache[t];
      launch_forward_pointwise(
          gates,
          c_prev,
          gate_t,
          cell_t,
          h_raw_t,
          torch::Tensor(),
          activation_mode);
      output[t].copy_(torch::mm(h_raw_t, w_hr_t));
    } else {
      launch_forward_pointwise(
          gates,
          c_prev,
          gate_t,
          cell_t,
          torch::Tensor(),
          output[t],
          activation_mode);
    }

    h_prev = output[t];
    c_prev = cell_t;
  }

  return {output, h_prev, c_prev, gate_cache, cell_cache, hidden_raw_cache};
}

std::vector<torch::Tensor> lstm_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& grad_h_n,
    const torch::Tensor& grad_c_n,
    const torch::Tensor& input,
    const torch::Tensor& output,
    const torch::Tensor& h0,
    const torch::Tensor& c0,
    const torch::Tensor& w_ih,
    const torch::Tensor& w_hh,
    const torch::Tensor& b_ih,
    const torch::Tensor& b_hh,
    const torch::Tensor& w_hr,
    const torch::Tensor& gate_cache,
    const torch::Tensor& cell_cache,
    const torch::Tensor& hidden_raw_cache,
    bool reverse,
    int64_t activation_mode) {
  check_activation_mode(activation_mode);
  TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
  if (activation_mode == kActivationMoffett) {
    TORCH_CHECK(
        grad_output.scalar_type() == torch::kBFloat16,
        "moffett activation expects torch.bfloat16 tensors");
  }

  c10::cuda::CUDAGuard device_guard(grad_output.device());

  auto grad_output_c = grad_output.contiguous();
  auto grad_h_next = grad_h_n.contiguous();
  auto grad_c_next = grad_c_n.contiguous();
  auto input_c = input.contiguous();
  auto output_c = output.contiguous();
  auto h0_c = h0.contiguous();
  auto c0_c = c0.contiguous();
  auto w_ih_c = w_ih.contiguous();
  auto w_hh_c = w_hh.contiguous();
  const auto use_proj = has_projection(w_hr);
  auto w_hr_c = use_proj ? w_hr.contiguous() : torch::Tensor();
  auto gate_cache_c = gate_cache.contiguous();
  auto cell_cache_c = cell_cache.contiguous();
  auto hidden_raw_cache_c =
      use_proj ? hidden_raw_cache.contiguous() : torch::Tensor();

  const auto seq_len = input_c.size(0);
  const auto batch = input_c.size(1);
  const auto input_size = input_c.size(2);
  const auto hidden = c0_c.size(1);
  const auto real_hidden = h0_c.size(1);

  auto options = input_c.options();
  auto prev_hidden_cache =
      reconstruct_prev_hidden(output_c, h0_c, reverse).contiguous();
  auto grad_gates = torch::empty({seq_len, batch, 4 * hidden}, options);
  auto grad_w_hr =
      use_proj ? torch::zeros_like(w_hr_c) : torch::empty({0, 0}, options);
  auto dy_cache = use_proj
      ? torch::empty({seq_len, batch, real_hidden}, options)
      : torch::Tensor();

  for (int64_t step = seq_len; step > 0; --step) {
    const auto t = time_index(step - 1, seq_len, reverse);
    auto c_prev =
        reverse ? (t == seq_len - 1 ? c0_c : cell_cache_c[t + 1])
                : (t == 0 ? c0_c : cell_cache_c[t - 1]);
    auto dy = grad_output_c[t] + grad_h_next;
    auto grad_h_raw = dy;
    if (use_proj) {
      dy_cache[t].copy_(dy);
      grad_h_raw = torch::mm(dy, w_hr_c);
    }

    launch_backward_pointwise(
        gate_cache_c[t],
        cell_cache_c[t],
        c_prev,
        grad_h_raw,
        grad_c_next,
        grad_gates[t],
        grad_c_next,
        activation_mode);
    grad_h_next = torch::mm(grad_gates[t], w_hh_c);
  }

  auto flat_grad_gates = grad_gates.reshape({seq_len * batch, 4 * hidden});
  auto flat_input = input_c.reshape({seq_len * batch, input_size});
  auto flat_prev_hidden =
      prev_hidden_cache.reshape({seq_len * batch, real_hidden});

  auto grad_input =
      torch::mm(flat_grad_gates, w_ih_c).reshape({seq_len, batch, input_size});
  auto grad_w_ih = torch::mm(flat_grad_gates.t(), flat_input);
  auto grad_w_hh = torch::mm(flat_grad_gates.t(), flat_prev_hidden);
  auto grad_bias = flat_grad_gates.sum(0);

  if (use_proj) {
    auto flat_dy = dy_cache.reshape({seq_len * batch, real_hidden});
    auto flat_hidden_raw = hidden_raw_cache_c.reshape({seq_len * batch, hidden});
    grad_w_hr = torch::mm(flat_dy.t(), flat_hidden_raw);
  }

  return {
      grad_input,
      grad_h_next,
      grad_c_next,
      grad_w_ih,
      grad_w_hh,
      grad_bias,
      grad_bias.clone(),
      grad_w_hr};
}
