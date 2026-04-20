#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <c10/macros/Macros.h>
#include <c10/util/BFloat16.h>

namespace moffett {

using BF16 = c10::BFloat16;
using U16 = uint16_t;
using U32 = uint32_t;

constexpr U16 BF16_NEG_INF  = 0xFF80;
constexpr U16 BF16_POS_INF  = 0x7F80;
constexpr U16 BF16_POS_ZERO = 0x0000;
constexpr U16 BF16_NEG_ZERO = 0x8000;
constexpr U16 BF16_NAN_VAL  = 0x7FC0;

constexpr int kTableCols = 5;
constexpr int kSigmoidRows = 12;
constexpr int kTanhRows = 8;
constexpr int kRcpRows = 4;

C10_HOST_DEVICE inline float u32_to_f32(U32 x) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(x);
#else
    float y;
    std::memcpy(&y, &x, sizeof(y));
    return y;
#endif
}

C10_HOST_DEVICE inline U32 f32_to_u32(float x) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(x);
#else
    U32 y;
    std::memcpy(&y, &x, sizeof(y));
    return y;
#endif
}

C10_HOST_DEVICE inline bool signbit_bf16(BF16 x) {
    return (x.x & 0x8000u) != 0;
}

C10_HOST_DEVICE inline bool signbit_f32(float x) {
    return (f32_to_u32(x) & 0x80000000u) != 0;
}

C10_HOST_DEVICE inline float mul20(float a, float b) {
    U32 v = f32_to_u32(a * b);
    v &= 0xfffff000u;
    return u32_to_f32(v);
}

C10_HOST_DEVICE inline float add20(float a, float b) {
    U32 v = f32_to_u32(a + b);
    v &= 0xfffff000u;
    return u32_to_f32(v);
}

C10_HOST_DEVICE inline float round16(float a) {
    U32 v = f32_to_u32(a);
    v &= 0xffff0000u;
    return u32_to_f32(v);
}

// Multiply a finite float by 2^n by adjusting the biased exponent. Falls back to
// ldexp for denormals / inf / nan / on overflow / underflow so the result still
// matches the original loop-based FloatShift behaviour.
C10_HOST_DEVICE inline float float_shift(float v, int n) {
    U32 u = f32_to_u32(v);
    const U32 sign = u & 0x80000000u;
    const U32 exp_biased = (u >> 23) & 0xffu;
    if (exp_biased == 0u || exp_biased == 0xffu) {
#ifdef __CUDA_ARCH__
        return ldexpf(v, n);
#else
        return std::ldexp(v, n);
#endif
    }
    const int new_exp = static_cast<int>(exp_biased) + n;
    if (new_exp >= 0xff) {
        return u32_to_f32(sign | 0x7f800000u);
    }
    if (new_exp <= 0) {
#ifdef __CUDA_ARCH__
        return ldexpf(v, n);
#else
        return std::ldexp(v, n);
#endif
    }
    const U32 mant = u & 0x007fffffu;
    return u32_to_f32(sign | (static_cast<U32>(new_exp) << 23) | mant);
}

// Flattened host-side tables. CUDA uses its own __constant__ copies in the .cu file.
static constexpr U32 kSigmoidHexHost[kSigmoidRows * kTableCols] = {
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

static constexpr U32 kTanhHexHost[kTanhRows * kTableCols] = {
    0x00000000,0x00800000,0x00000000,0x00000000,0x00000000,
    0x00800000,0x32000000,0x00000000,0x3f800000,0x00000000,
    0x32000000,0x3e000000,0x325f3000,0x3f800000,0x18906000,
    0x3e000000,0x3f000000,0xbe89a000,0x3f890000,0xbbb69000,
    0x3f000000,0x3fc00000,0xbe9ef000,0x3f87d000,0x3c586000,
    0x3fc00000,0x40200000,0xbd94e000,0x3ebe2000,0x3f038000,
    0x40200000,0x40500000,0xbc6f7000,0x3dc85000,0x3f558000,
    0x40500000,0x7f800000,0x00000000,0x00000000,0x3f800000,
};

// Reciprocal (1/x) approximation table. Inputs are first folded to [+1.0, +2.0)
// by sign-stripping and a power-of-two shift, then the per-interval polynomial
// a*x^2 + b*x + c is evaluated.
static constexpr U32 kRcpHexHost[kRcpRows * kTableCols] = {
    0x3f800000,0x3f900000,0x3f249000,0xc0101000,0x40270000,
    0x3f900000,0x3fc00000,0x3ef51000,0xbfeb6000,0x40167000,
    0x3fc00000,0x3fe00000,0x3e703000,0xbf924000,0x3fed3000,
    0x3fe00000,0x40000000,0x3e1c1000,0xbf5b6000,0x3fcd6000,
};

C10_HOST_DEVICE inline float table_f32(const U32* table, int row, int col) {
    return u32_to_f32(table[row * kTableCols + col]);
}

C10_HOST_DEVICE inline int find_sigmoid_interval(float x, const U32* table) {
    for (int itv = 0; itv < kSigmoidRows; ++itv) {
        const float left = table_f32(table, itv, 0);
        const float right = table_f32(table, itv, 1);
        if (left < 0.0f) {
            if (x > left && x <= right) {
                if (right == 0.0f && signbit_f32(right) != signbit_f32(x)) {
                    continue;
                }
                return itv;
            }
        } else {
            if (x >= left && x < right) {
                return itv;
            }
        }
    }
    return 0;
}

C10_HOST_DEVICE inline int find_tanh_interval(float x, const U32* table) {
    for (int itv = 0; itv < kTanhRows; ++itv) {
        if (x >= table_f32(table, itv, 0) && x < table_f32(table, itv, 1)) {
            return itv;
        }
    }
    return 0;
}

C10_HOST_DEVICE inline BF16 sigmoid_one_from_table(BF16 x_bf16, const U32* table) {
    if (x_bf16.x == BF16_NAN_VAL) {
        BF16 y = BF16(1.0f);
        y.x = BF16_NAN_VAL;
        return y;
    }
    if (x_bf16.x == BF16_POS_INF) {
        return BF16(1.0f);
    }
    if (x_bf16.x == BF16_NEG_INF) {
        BF16 y = BF16(1.0f);
        y.x = BF16_POS_ZERO;
        return y;
    }

    const float x = static_cast<float>(x_bf16);
    const int interval = find_sigmoid_interval(x, table);

    float partial1 = mul20(x, x);
    float partial2 = mul20(table_f32(table, interval, 3), x);
    partial1 = mul20(table_f32(table, interval, 2), partial1);
    partial2 = add20(table_f32(table, interval, 4), partial2);
    return BF16(add20(partial2, partial1));
}

C10_HOST_DEVICE inline BF16 tanh_one_from_table(BF16 x_bf16, const U32* table) {
    if (x_bf16.x == BF16_NAN_VAL) {
        BF16 y = BF16(1.0f);
        y.x = BF16_NAN_VAL;
        return y;
    }
    if (x_bf16.x == BF16_POS_INF) {
        return BF16(1.0f);
    }
    if (x_bf16.x == BF16_NEG_INF) {
        return BF16(-1.0f);
    }

    float symbol = 1.0f;
    if (signbit_bf16(x_bf16)) {
        symbol = -1.0f;
        BF16 tmp = BF16(0.0f);
        tmp.x = static_cast<U16>(x_bf16.x & 0x7fffu);
        x_bf16 = tmp;
    }

    const float x = static_cast<float>(x_bf16);
    const int interval = find_tanh_interval(x, table);

    float partial1 = mul20(x, x);
    float partial2 = mul20(table_f32(table, interval, 3), x);
    partial1 = mul20(table_f32(table, interval, 2), partial1);
    partial2 = add20(table_f32(table, interval, 4), partial2);
    return BF16(symbol * add20(partial1, partial2));
}

C10_HOST_DEVICE inline BF16 sigmoid_one_cpu(BF16 x_bf16) {
    return sigmoid_one_from_table(x_bf16, kSigmoidHexHost);
}

C10_HOST_DEVICE inline BF16 tanh_one_cpu(BF16 x_bf16) {
    return tanh_one_from_table(x_bf16, kTanhHexHost);
}

C10_HOST_DEVICE inline int find_rcp_interval(float x, const U32* table) {
    for (int itv = 0; itv < kRcpRows; ++itv) {
        const float left = table_f32(table, itv, 0);
        const float right = table_f32(table, itv, 1);
        if (x >= left && x < right) {
            return itv;
        }
    }
    return kRcpRows - 1;
}

C10_HOST_DEVICE inline BF16 rcp_one_from_table(BF16 x_bf16, const U32* table) {
    if (x_bf16.x == BF16_NAN_VAL) {
        BF16 y = BF16(1.0f);
        y.x = BF16_NAN_VAL;
        return y;
    }

    float symbol = 1.0f;
    if (signbit_bf16(x_bf16)) {
        symbol = -1.0f;
        BF16 tmp = BF16(0.0f);
        tmp.x = static_cast<U16>(x_bf16.x & 0x7fffu);
        x_bf16 = tmp;
    }

    // 1/(+inf) -> +0, 1/(-inf) -> -0
    if (x_bf16.x == BF16_POS_INF) {
        BF16 y = BF16(0.0f);
        y.x = (symbol < 0.0f) ? BF16_NEG_ZERO : BF16_POS_ZERO;
        return y;
    }

    // 1/(+0) -> +inf, 1/(-0) -> -inf
    if (x_bf16.x == BF16_POS_ZERO) {
        BF16 y = BF16(0.0f);
        y.x = (symbol < 0.0f) ? BF16_NEG_INF : BF16_POS_INF;
        return y;
    }

    float x = static_cast<float>(x_bf16);

    // Normalise x into [1.0, 2.0) by power-of-two shifting; mirrors the
    // shift_bit == 1 branch of the reference ShiftActiv loop. For normal
    // floats we can derive the shift directly from the exponent in one step;
    // denormals fall back to the iterative form for correctness.
    int shift = 0;
    constexpr float kLeft = 1.0f;
    constexpr float kRight = 2.0f;
    {
        const U32 ux = f32_to_u32(x);
        const U32 exp_biased = (ux >> 23) & 0xffu;
        if (exp_biased != 0u && exp_biased != 0xffu) {
            const int e = static_cast<int>(exp_biased) - 127;
            shift = -e;
            const U32 mant = ux & 0x007fffffu;
            x = u32_to_f32((127u << 23) | mant);
        } else {
            int guard = 0;
            while (!(x >= kLeft && x < kRight)) {
                if (x < kLeft) {
                    shift += 1;
                    x = float_shift(x, 1);
                } else {
                    shift -= 1;
                    x = float_shift(x, -1);
                }
                if (++guard > 300) {
                    break;
                }
            }
        }
    }

    const int interval = find_rcp_interval(x, table);

    float partial1 = mul20(x, x);
    float partial2 = mul20(table_f32(table, interval, 3), x);
    partial1 = mul20(table_f32(table, interval, 2), partial1);
    partial2 = add20(table_f32(table, interval, 4), partial2);
    float result = add20(partial1, partial2);

    result = float_shift(result, shift) * symbol;
    return BF16(result);
}

C10_HOST_DEVICE inline BF16 rcp_one_cpu(BF16 x_bf16) {
    return rcp_one_from_table(x_bf16, kRcpHexHost);
}

} // namespace moffett
