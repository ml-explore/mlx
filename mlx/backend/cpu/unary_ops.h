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
    return 1.0f / (1.0f + simd::exp(-x));
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

} // namespace mlx::core::detail
