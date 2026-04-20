#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include "util_common.hpp"

namespace moffett {

__device__ __constant__ U32 kSigmoidHexDev[kSigmoidRows * kTableCols] = {
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

__device__ __constant__ U32 kTanhHexDev[kTanhRows * kTableCols] = {
    0x00000000,0x00800000,0x00000000,0x00000000,0x00000000,
    0x00800000,0x32000000,0x00000000,0x3f800000,0x00000000,
    0x32000000,0x3e000000,0x325f3000,0x3f800000,0x18906000,
    0x3e000000,0x3f000000,0xbe89a000,0x3f890000,0xbbb69000,
    0x3f000000,0x3fc00000,0xbe9ef000,0x3f87d000,0x3c586000,
    0x3fc00000,0x40200000,0xbd94e000,0x3ebe2000,0x3f038000,
    0x40200000,0x40500000,0xbc6f7000,0x3dc85000,0x3f558000,
    0x40500000,0x7f800000,0x00000000,0x00000000,0x3f800000,
};

__device__ __constant__ U32 kRcpHexDev[kRcpRows * kTableCols] = {
    0x3f800000,0x3f900000,0x3f249000,0xc0101000,0x40270000,
    0x3f900000,0x3fc00000,0x3ef51000,0xbfeb6000,0x40167000,
    0x3fc00000,0x3fe00000,0x3e703000,0xbf924000,0x3fed3000,
    0x3fe00000,0x40000000,0x3e1c1000,0xbf5b6000,0x3fcd6000,
};

namespace {

enum class UnaryOp { Sigmoid, Tanh, Rcp };

__device__ inline BF16 sigmoid_one_cuda(BF16 x) {
    return sigmoid_one_from_table(x, kSigmoidHexDev);
}

__device__ inline BF16 tanh_one_cuda(BF16 x) {
    return tanh_one_from_table(x, kTanhHexDev);
}

__device__ inline BF16 rcp_one_cuda(BF16 x) {
    return rcp_one_from_table(x, kRcpHexDev);
}

template <UnaryOp kOp>
__global__ void unary_kernel(const BF16* in, BF16* out, int64_t n) {
    const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    while (idx < n) {
        if constexpr (kOp == UnaryOp::Sigmoid) {
            out[idx] = sigmoid_one_cuda(in[idx]);
        } else if constexpr (kOp == UnaryOp::Tanh) {
            out[idx] = tanh_one_cuda(in[idx]);
        } else {
            out[idx] = rcp_one_cuda(in[idx]);
        }
        idx += stride;
    }
}

template <UnaryOp kOp>
inline void launch_unary(const BF16* in, BF16* out, int64_t n) {
    if (n == 0) {
        return;
    }
    constexpr int threads = 256;
    const int64_t blocks64 = (n + threads - 1) / threads;
    const int blocks = static_cast<int>(blocks64 > 4096 ? 4096 : blocks64);
    auto stream = at::cuda::getCurrentCUDAStream();
    unary_kernel<kOp><<<blocks, threads, 0, stream>>>(in, out, n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace

void sigmoid_forward_cuda(const BF16* in, BF16* out, int64_t n) {
    launch_unary<UnaryOp::Sigmoid>(in, out, n);
}

void tanh_forward_cuda(const BF16* in, BF16* out, int64_t n) {
    launch_unary<UnaryOp::Tanh>(in, out, n);
}

void rcp_forward_cuda(const BF16* in, BF16* out, int64_t n) {
    launch_unary<UnaryOp::Rcp>(in, out, n);
}

} // namespace moffett
