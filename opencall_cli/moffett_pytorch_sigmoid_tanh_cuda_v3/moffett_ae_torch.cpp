#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include "util_common.hpp"

namespace moffett {
void sigmoid_forward_cpu(const BF16* in, BF16* out, int64_t n);
void tanh_forward_cpu(const BF16* in, BF16* out, int64_t n);
void sigmoid_forward_cuda(const BF16* in, BF16* out, int64_t n);
void tanh_forward_cuda(const BF16* in, BF16* out, int64_t n);
}

namespace {

torch::Tensor run_impl(torch::Tensor x, bool use_sigmoid) {
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "input must be torch.bfloat16");

    auto x_contig = x.contiguous();
    auto flat_in = x_contig.view({-1});
    auto flat_out = torch::empty_like(flat_in);

    const auto* in_ptr = reinterpret_cast<const moffett::BF16*>(flat_in.data_ptr<c10::BFloat16>());
    auto* out_ptr = reinterpret_cast<moffett::BF16*>(flat_out.data_ptr<c10::BFloat16>());
    const int64_t numel = flat_in.numel();

    if (x_contig.is_cuda()) {
        c10::cuda::CUDAGuard device_guard(x_contig.device());
        if (use_sigmoid) {
            moffett::sigmoid_forward_cuda(in_ptr, out_ptr, numel);
        } else {
            moffett::tanh_forward_cuda(in_ptr, out_ptr, numel);
        }
    } else {
        constexpr int64_t kGrainSize = 32768;
        at::parallel_for(0, numel, kGrainSize, [&](int64_t begin, int64_t end) {
            const int64_t len = end - begin;
            if (use_sigmoid) {
                moffett::sigmoid_forward_cpu(in_ptr + begin, out_ptr + begin, len);
            } else {
                moffett::tanh_forward_cpu(in_ptr + begin, out_ptr + begin, len);
            }
        });
    }

    return flat_out.view(x.sizes());
}

} // namespace

torch::Tensor moffett_sigmoid_forward(torch::Tensor x) {
    return run_impl(std::move(x), true);
}

torch::Tensor moffett_tanh_forward(torch::Tensor x) {
    return run_impl(std::move(x), false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sigmoid_forward", &moffett_sigmoid_forward, "Moffett-style sigmoid forward (CPU/CUDA, bf16)");
    m.def("tanh_forward", &moffett_tanh_forward, "Moffett-style tanh forward (CPU/CUDA, bf16)");
}
