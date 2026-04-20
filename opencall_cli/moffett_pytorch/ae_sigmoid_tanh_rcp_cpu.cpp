#include <ATen/Parallel.h>
#include <torch/extension.h>
#include "util_common.hpp"

namespace moffett {

void sigmoid_forward_cpu(const BF16* in, BF16* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = sigmoid_one_cpu(in[i]);
    }
}

void tanh_forward_cpu(const BF16* in, BF16* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = tanh_one_cpu(in[i]);
    }
}

void rcp_forward_cpu(const BF16* in, BF16* out, int64_t n) {
    for (int64_t i = 0; i < n; ++i) {
        out[i] = rcp_one_cpu(in[i]);
    }
}

} // namespace moffett
