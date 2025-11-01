// Copyright © 2024 Apple Inc.

#pragma once

#include <stdint.h>
#include <cmath>
#include <complex>

#include "mlx/backend/cpu/simd/simd.h"

namespace mlx::core::detail {

using namespace mlx::core::simd;

#define SINGLE()                         \
  template <typename T>                  \
  T operator()(T x) {                    \
    return (*this)(Simd<T, 1>(x)).value; \
  }

#define DEFAULT_OP(Op, op)                \
  struct Op {                             \
    template <int N, typename T>          \
    Simd<T, N> operator()(Simd<T, N> x) { \
      return simd::op(x);                 \
    }                                     \
    SINGLE()                              \
  };

DEFAULT_OP(Abs, abs)
DEFAULT_OP(ArcCos, acos)
DEFAULT_OP(ArcCosh, acosh)
DEFAULT_OP(ArcSin, asin)
DEFAULT_OP(ArcSinh, asinh)
DEFAULT_OP(ArcTan, atan)
DEFAULT_OP(ArcTanh, atanh)
DEFAULT_OP(BitwiseInvert, operator~)
DEFAULT_OP(Ceil, ceil)
DEFAULT_OP(Conjugate, conj)
DEFAULT_OP(Cos, cos)
DEFAULT_OP(Cosh, cosh)
DEFAULT_OP(Erf, erf)
DEFAULT_OP(ErfInv, erfinv)
DEFAULT_OP(Exp, exp)
DEFAULT_OP(Expm1, expm1)
DEFAULT_OP(Floor, floor);
DEFAULT_OP(Log, log);
DEFAULT_OP(Log2, log2);
DEFAULT_OP(Log10, log10);
DEFAULT_OP(Log1p, log1p);
DEFAULT_OP(LogicalNot, operator!)
DEFAULT_OP(Negative, operator-)
DEFAULT_OP(Round, rint);
DEFAULT_OP(Sin, sin)
DEFAULT_OP(Sinh, sinh)
DEFAULT_OP(Sqrt, sqrt)
DEFAULT_OP(Rsqrt, rsqrt)
DEFAULT_OP(Tan, tan)
DEFAULT_OP(Tanh, tanh)

struct Imag {
  template <int N>
  Simd<float, N> operator()(Simd<complex64_t, N> x) {
    return simd::imag(x);
  }
  SINGLE()
};

struct Real {
  template <int N>
  Simd<float, N> operator()(Simd<complex64_t, N> x) {
    return simd::real(x);
  }
  SINGLE()
};

struct Sigmoid {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x) {
    auto y = 1.0f / (1.0f + simd::exp(simd::abs(x)));
    return simd::select(x < Simd<T, N>{0}, y, Simd<T, N>{1} - y);
  }
  SINGLE()
};

struct Sign {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x) {
    auto z = Simd<T, N>{0};
    auto o = Simd<T, N>{1};
    auto m = Simd<T, N>{-1};
    if constexpr (std::is_unsigned_v<T>) {
      return simd::select(x == z, z, o);
    } else if constexpr (std::is_same_v<T, complex64_t>) {
      return simd::select(x == z, x, Simd<T, N>(x / simd::abs(x)));
    } else {
      return simd::select(x < z, m, simd::select(x > z, o, z));
    }
  }
  SINGLE()
};

struct Square {
  template <int N, typename T>
  Simd<T, N> operator()(Simd<T, N> x) {
    return x * x;
  }
  SINGLE()
};

template <int N>
Simd<float, N> fp32_from_bits(Simd<uint32_t, N> x) {
  return *(Simd<float, N>*)(&x);
}
template <int N>
Simd<uint32_t, N> fp32_to_bits(Simd<float, N> x) {
  return *(Simd<uint32_t, N>*)(&x);
}

struct ToFP8 {
  template <typename T, int N>
  Simd<uint8_t, N> operator()(Simd<T, N> f) {
    uint32_t fp8_max = 543 << 21;
    auto denorm_mask = Simd<uint32_t, N>(141 << 23);
    Simd<uint32_t, N> f_bits;
    Simd<float, N> f32 = f;
    f_bits = fp32_to_bits(f32);
    Simd<uint8_t, N> result = 0u;
    auto sign = f_bits & 0x80000000;
    f_bits = f_bits ^ sign;

    auto f_bits_low =
        fp32_to_bits(fp32_from_bits(f_bits) + fp32_from_bits(denorm_mask));
    auto result_low = Simd<uint8_t, N>(f_bits_low - denorm_mask);

    auto mant_odd = Simd<uint8_t, N>((f_bits >> 20) & 1);
    auto f_bits_high = f_bits + (((uint32_t)(7 - 127) << 23) + 0x7FFFF);
    f_bits_high = f_bits_high + Simd<uint32_t, N>(mant_odd);

    auto result_high = Simd<uint8_t, N>(f_bits_high >> 20);
    result = select(f_bits < (121 << 23), result_low, result_high);

    auto result_sat = Simd<uint8_t, N>(0x7E);
    result = select(f_bits >= fp8_max, result_sat, result);
    return result | Simd<uint8_t, N>(sign >> 24);
  }

  template <typename T>
  uint8_t operator()(T x) {
    return (*this)(Simd<T, 1>(x)).value;
  }
};

struct FromFP8 {
  template <int N>
  Simd<float, N> operator()(Simd<uint8_t, N> x) {
    auto w = Simd<uint32_t, N>(x) << 24;
    auto sign = w & 0x80000000;
    auto nonsign = w & 0x7FFFFFFF;

    auto renorm_shift = clz(nonsign);
    renorm_shift = simd::select(
        renorm_shift > Simd<uint32_t, N>{4},
        renorm_shift - Simd<uint32_t, N>{4},
        Simd<uint32_t, N>{0});

    Simd<int32_t, N> inf_nan_mask =
        (Simd<int32_t, N>(nonsign + 0x01000000) >> 8) & 0x7F800000;
    auto zero_mask = Simd<int32_t, N>(nonsign - 1) >> 31;
    auto result = sign |
        ((((nonsign << renorm_shift >> 4) + ((0x78 - renorm_shift) << 23)) |
          inf_nan_mask) &
         ~zero_mask);
    return fp32_from_bits(result);
  }
  float operator()(uint8_t x) {
    return (*this)(Simd<uint8_t, 1>(x)).value;
  }
};
} // namespace mlx::core::detail
