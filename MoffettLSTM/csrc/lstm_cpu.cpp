#include <ATen/Parallel.h>
#include <torch/extension.h>

#include <array>
#include <vector>

#include "util_common.hpp"

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
    int64_t activation_mode);

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
    int64_t activation_mode);

namespace {

constexpr int64_t kActivationNative = 0;
constexpr int64_t kActivationCustom = 1;
constexpr int64_t kActivationFormulaFp64 = 2;
constexpr int64_t kActivationMoffett = 3;

void check_tensor(
    const torch::Tensor& tensor,
    const char* name,
    int64_t expected_dim,
    bool allow_empty = false) {
  TORCH_CHECK(tensor.defined(), name, " must be defined");
  TORCH_CHECK(tensor.is_floating_point(), name, " must be floating point");
  TORCH_CHECK(
      allow_empty || tensor.numel() > 0, name, " must not be empty");
  TORCH_CHECK(
      tensor.dim() == expected_dim,
      name,
      " must be ",
      expected_dim,
      "D, got ",
      tensor.dim(),
      "D");
}

void check_same_meta(
    const torch::Tensor& tensor,
    const torch::Tensor& reference,
    const char* name) {
  TORCH_CHECK(
      tensor.device() == reference.device(),
      name,
      " must be on device ",
      reference.device(),
      ", got ",
      tensor.device());
  TORCH_CHECK(
      tensor.scalar_type() == reference.scalar_type(),
      name,
      " must have dtype ",
      reference.scalar_type(),
      ", got ",
      tensor.scalar_type());
}

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

torch::Tensor moffett_unary_cpu(
    const torch::Tensor& x,
    bool use_sigmoid) {
  TORCH_CHECK(
      x.scalar_type() == torch::kBFloat16,
      "moffett activation expects torch.bfloat16 input");
  auto x_contig = x.contiguous();
  auto flat_in = x_contig.reshape({-1});
  auto flat_out = torch::empty_like(flat_in);

  const auto* in_ptr =
      reinterpret_cast<const moffett::BF16*>(flat_in.data_ptr<c10::BFloat16>());
  auto* out_ptr =
      reinterpret_cast<moffett::BF16*>(flat_out.data_ptr<c10::BFloat16>());
  const auto n = flat_in.numel();
  constexpr int64_t kGrainSize = 32768;
  at::parallel_for(0, n, kGrainSize, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      out_ptr[i] = use_sigmoid ? moffett::sigmoid_one_cpu(in_ptr[i])
                               : moffett::tanh_one_cpu(in_ptr[i]);
    }
  });
  return flat_out.view(x.sizes());
}

torch::Tensor sigmoid_formula_fp64(const torch::Tensor& x) {
  auto x64 = x.to(torch::kFloat64);
  auto positive = x64 >= 0;
  auto z_pos = torch::exp(-x64);
  auto z_neg = torch::exp(x64);
  auto y64 =
      torch::where(positive, 1.0 / (1.0 + z_pos), z_neg / (1.0 + z_neg));
  return y64.to(x.scalar_type());
}

torch::Tensor tanh_formula_fp64(const torch::Tensor& x) {
  auto x64 = x.to(torch::kFloat64);
  auto z = torch::exp(-2.0 * torch::abs(x64));
  auto y64 = torch::sign(x64) * ((1.0 - z) / (1.0 + z));
  return y64.to(x.scalar_type());
}

torch::Tensor apply_sigmoid(const torch::Tensor& x, int64_t activation_mode) {
  if (activation_mode == kActivationMoffett) {
    return moffett_unary_cpu(x, true);
  }
  if (activation_mode == kActivationNative ||
      activation_mode == kActivationCustom) {
    return torch::sigmoid(x);
  }
  return sigmoid_formula_fp64(x);
}

torch::Tensor apply_tanh(const torch::Tensor& x, int64_t activation_mode) {
  if (activation_mode == kActivationMoffett) {
    return moffett_unary_cpu(x, false);
  }
  if (activation_mode == kActivationNative ||
      activation_mode == kActivationCustom) {
    return torch::tanh(x);
  }
  return tanh_formula_fp64(x);
}

void check_projection_weight(const torch::Tensor& weight_hr) {
  if (!has_projection(weight_hr)) {
    return;
  }
  check_tensor(weight_hr, "w_hr", 2);
}

std::array<torch::Tensor, 4> split_gates(
    const torch::Tensor& gates,
    int64_t hidden) {
  return {
      gates.slice(1, 0, hidden),
      gates.slice(1, hidden, 2 * hidden),
      gates.slice(1, 2 * hidden, 3 * hidden),
      gates.slice(1, 3 * hidden, 4 * hidden)};
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

std::vector<torch::Tensor> lstm_forward_cpu(
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
  check_tensor(input, "input", 3);
  check_tensor(h0, "h0", 2);
  check_tensor(c0, "c0", 2);
  check_tensor(w_ih, "w_ih", 2);
  check_tensor(w_hh, "w_hh", 2);
  check_tensor(b_ih, "b_ih", 1, true);
  check_tensor(b_hh, "b_hh", 1, true);
  check_projection_weight(w_hr);
  check_same_meta(h0, input, "h0");
  check_same_meta(c0, input, "c0");
  check_same_meta(w_ih, input, "w_ih");
  check_same_meta(w_hh, input, "w_hh");
  check_same_meta(b_ih, input, "b_ih");
  check_same_meta(b_hh, input, "b_hh");
  if (has_projection(w_hr)) {
    check_same_meta(w_hr, input, "w_hr");
  }

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

  TORCH_CHECK(h_prev.size(0) == batch, "h0 batch dimension mismatch");
  TORCH_CHECK(c_prev.size(0) == batch, "c0 batch dimension mismatch");
  TORCH_CHECK(w_ih_c.size(0) == 4 * hidden, "w_ih first dimension mismatch");
  TORCH_CHECK(w_ih_c.size(1) == input_size, "w_ih second dimension mismatch");
  TORCH_CHECK(w_hh_c.size(0) == 4 * hidden, "w_hh first dimension mismatch");
  TORCH_CHECK(
      w_hh_c.size(1) == real_hidden, "w_hh second dimension mismatch");
  TORCH_CHECK(b_total.size(0) == 4 * hidden, "bias dimension mismatch");

  if (use_proj) {
    TORCH_CHECK(
        w_hr_c.size(0) == real_hidden, "w_hr first dimension mismatch");
    TORCH_CHECK(
        w_hr_c.size(1) == hidden, "w_hr second dimension mismatch");
  } else {
    TORCH_CHECK(
        real_hidden == hidden,
        "Without projection, hidden state size must equal cell size");
  }

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
    auto chunks = split_gates(gates, hidden);

    auto i = apply_sigmoid(chunks[0], activation_mode);
    auto f = apply_sigmoid(chunks[1], activation_mode);
    auto g = apply_tanh(chunks[2], activation_mode);
    auto o = apply_sigmoid(chunks[3], activation_mode);

    auto c_t = f * c_prev + i * g;
    auto h_raw = o * apply_tanh(c_t, activation_mode);
    auto h_t = use_proj ? torch::mm(h_raw, w_hr_t) : h_raw;

    gate_cache[t].slice(1, 0, hidden).copy_(i);
    gate_cache[t].slice(1, hidden, 2 * hidden).copy_(f);
    gate_cache[t].slice(1, 2 * hidden, 3 * hidden).copy_(g);
    gate_cache[t].slice(1, 3 * hidden, 4 * hidden).copy_(o);
    cell_cache[t].copy_(c_t);
    if (use_proj) {
      hidden_raw_cache[t].copy_(h_raw);
    }
    output[t].copy_(h_t);

    h_prev = h_t;
    c_prev = c_t;
  }

  return {output, h_prev, c_prev, gate_cache, cell_cache, hidden_raw_cache};
}

std::vector<torch::Tensor> lstm_backward_cpu(
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
  check_tensor(grad_output, "grad_output", 3);
  check_tensor(grad_h_n, "grad_h_n", 2);
  check_tensor(grad_c_n, "grad_c_n", 2);
  check_tensor(input, "input", 3);
  check_tensor(output, "output", 3);
  check_tensor(h0, "h0", 2);
  check_tensor(c0, "c0", 2);
  check_tensor(w_ih, "w_ih", 2);
  check_tensor(w_hh, "w_hh", 2);
  check_tensor(b_ih, "b_ih", 1, true);
  check_tensor(b_hh, "b_hh", 1, true);
  check_projection_weight(w_hr);
  check_tensor(gate_cache, "gate_cache", 3);
  check_tensor(cell_cache, "cell_cache", 3);
  check_tensor(hidden_raw_cache, "hidden_raw_cache", 3, true);
  check_same_meta(grad_h_n, grad_output, "grad_h_n");
  check_same_meta(grad_c_n, grad_output, "grad_c_n");
  check_same_meta(input, grad_output, "input");
  check_same_meta(output, grad_output, "output");
  check_same_meta(h0, grad_output, "h0");
  check_same_meta(c0, grad_output, "c0");
  check_same_meta(w_ih, grad_output, "w_ih");
  check_same_meta(w_hh, grad_output, "w_hh");
  check_same_meta(b_ih, grad_output, "b_ih");
  check_same_meta(b_hh, grad_output, "b_hh");
  check_same_meta(gate_cache, grad_output, "gate_cache");
  check_same_meta(cell_cache, grad_output, "cell_cache");
  const auto use_proj = has_projection(w_hr);
  if (use_proj) {
    check_same_meta(w_hr, grad_output, "w_hr");
    check_same_meta(hidden_raw_cache, grad_output, "hidden_raw_cache");
  }

  auto grad_output_c = grad_output.contiguous();
  auto grad_h_next = grad_h_n.contiguous();
  auto grad_c_next = grad_c_n.contiguous();
  auto input_c = input.contiguous();
  auto output_c = output.contiguous();
  auto h0_c = h0.contiguous();
  auto c0_c = c0.contiguous();
  auto w_ih_c = w_ih.contiguous();
  auto w_hh_c = w_hh.contiguous();
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

  TORCH_CHECK(
      grad_output_c.size(0) == seq_len && grad_output_c.size(1) == batch &&
          grad_output_c.size(2) == real_hidden,
      "grad_output shape mismatch");
  TORCH_CHECK(
      grad_h_next.size(0) == batch && grad_h_next.size(1) == real_hidden,
      "grad_h_n shape mismatch");
  TORCH_CHECK(
      grad_c_next.size(0) == batch && grad_c_next.size(1) == hidden,
      "grad_c_n shape mismatch");

  auto options = input_c.options();
  auto prev_hidden_cache_c =
      reconstruct_prev_hidden(output_c, h0_c, reverse).contiguous();
  auto grad_gates = torch::empty({seq_len, batch, 4 * hidden}, options);
  auto grad_w_hr =
      use_proj ? torch::zeros_like(w_hr_c) : torch::empty({0, 0}, options);

  for (int64_t step = seq_len; step > 0; --step) {
    const auto t = time_index(step - 1, seq_len, reverse);

    auto i = gate_cache_c[t].slice(1, 0, hidden);
    auto f = gate_cache_c[t].slice(1, hidden, 2 * hidden);
    auto g = gate_cache_c[t].slice(1, 2 * hidden, 3 * hidden);
    auto o = gate_cache_c[t].slice(1, 3 * hidden, 4 * hidden);
    auto c_t = cell_cache_c[t];
    auto c_prev =
        reverse ? (t == seq_len - 1 ? c0_c : cell_cache_c[t + 1])
                : (t == 0 ? c0_c : cell_cache_c[t - 1]);

    auto dy = grad_output_c[t] + grad_h_next;
    auto dh = dy;
    if (use_proj) {
      auto h_raw = hidden_raw_cache_c[t];
      grad_w_hr += torch::mm(dy.t(), h_raw);
      dh = torch::mm(dy, w_hr_c);
    }

    auto tanh_c = apply_tanh(c_t, activation_mode);
    auto dc = grad_c_next + dh * o * (1 - tanh_c * tanh_c);

    auto d_o = dh * tanh_c;
    auto d_i = dc * g;
    auto d_f = dc * c_prev;
    auto d_g = dc * i;

    auto d_o_pre = d_o * o * (1 - o);
    auto d_i_pre = d_i * i * (1 - i);
    auto d_f_pre = d_f * f * (1 - f);
    auto d_g_pre = d_g * (1 - g * g);

    auto d_cat = torch::cat({d_i_pre, d_f_pre, d_g_pre, d_o_pre}, 1);
    grad_gates[t].copy_(d_cat);

    grad_h_next = torch::mm(d_cat, w_hh_c);
    grad_c_next = dc * f;
  }

  auto flat_grad_gates = grad_gates.reshape({seq_len * batch, 4 * hidden});
  auto flat_input = input_c.reshape({seq_len * batch, input_size});
  auto flat_prev_hidden =
      prev_hidden_cache_c.reshape({seq_len * batch, real_hidden});

  auto grad_input =
      torch::mm(flat_grad_gates, w_ih_c).reshape({seq_len, batch, input_size});
  auto grad_w_ih = torch::mm(flat_grad_gates.t(), flat_input);
  auto grad_w_hh = torch::mm(flat_grad_gates.t(), flat_prev_hidden);
  auto grad_bias = flat_grad_gates.sum(0);

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

std::vector<torch::Tensor> lstm_forward_dispatch(
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
  if (input.is_cuda()) {
#ifdef WITH_CUDA
    return lstm_forward_cuda(
        input,
        h0,
        c0,
        w_ih,
        w_hh,
        b_ih,
        b_hh,
        w_hr,
        reverse,
        activation_mode);
#else
    TORCH_CHECK(false, "CUDA tensors require building the CUDA extension");
#endif
  }
  return lstm_forward_cpu(
      input,
      h0,
      c0,
      w_ih,
      w_hh,
      b_ih,
      b_hh,
      w_hr,
      reverse,
      activation_mode);
}

std::vector<torch::Tensor> lstm_backward_dispatch(
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
  if (grad_output.is_cuda()) {
#ifdef WITH_CUDA
    return lstm_backward_cuda(
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
        reverse,
        activation_mode);
#else
    TORCH_CHECK(false, "CUDA tensors require building the CUDA extension");
#endif
  }
  return lstm_backward_cpu(
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
      reverse,
      activation_mode);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lstm_forward", &lstm_forward_dispatch, "Fast LSTM cell forward");
  m.def("lstm_backward", &lstm_backward_dispatch, "Fast LSTM cell backward");
}
