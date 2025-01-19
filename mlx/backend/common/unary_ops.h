// Copyright © 2024 Apple Inc.

#pragma once

#include <stdint.h>
#include <cmath>
#include <complex>

#include "mlx/backend/common/math.h"
#include "mlx/types/complex.h"

namespace mlx::core::detail {

#define DEFAULT_LOOPED()                        \
  template <typename T, typename U = T>         \
  void operator()(const T* x, U* y, size_t n) { \
    for (size_t i = 0; i < n; ++i) {            \
      y[i] = (*this)(x[i]);                     \
    }                                           \
  }

struct Abs {
  template <typename T>
  T operator()(T x) {
    return std::abs(x);
  }

  DEFAULT_LOOPED()
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return std::acos(x);
  }
  DEFAULT_LOOPED()
};

struct ArcCosh {
  template <typename T>
  T operator()(T x) {
    return std::acosh(x);
  }
  DEFAULT_LOOPED()
};

struct ArcSin {
  template <typename T>
  T operator()(T x) {
    return std::asin(x);
  }
  DEFAULT_LOOPED()
};

struct ArcSinh {
  template <typename T>
  T operator()(T x) {
    return std::asinh(x);
  }
  DEFAULT_LOOPED()
};

struct ArcTan {
  template <typename T>
  T operator()(T x) {
    return std::atan(x);
  }
  DEFAULT_LOOPED()
};

struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return std::atanh(x);
  }
  DEFAULT_LOOPED()
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return std::ceil(x);
  }
  DEFAULT_LOOPED()
};

struct Conjugate {
  complex64_t operator()(complex64_t x) {
    return std::conj(x);
  }
  DEFAULT_LOOPED()
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return std::cos(x);
  }
  DEFAULT_LOOPED()
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return std::cosh(x);
  }
  DEFAULT_LOOPED()
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erf(static_cast<float>(x)));
  }
  DEFAULT_LOOPED()
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erfinv(static_cast<float>(x)));
  }
  DEFAULT_LOOPED()
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return fast_exp(x);
  }

  complex64_t operator()(complex64_t x) {
    return std::exp(x);
  }
  DEFAULT_LOOPED()
};

struct Expm1 {
  template <typename T>
  T operator()(T x) {
    return expm1(x);
  }
  DEFAULT_LOOPED()
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return std::floor(x);
  }
  DEFAULT_LOOPED()
};

struct Imag {
  template <typename T>
  T operator()(T x) {
    return std::imag(x);
  }
  DEFAULT_LOOPED()
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return std::log(x);
  }
  DEFAULT_LOOPED()
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return std::log2(x);
  }
  DEFAULT_LOOPED()
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return std::log10(x);
  }
  DEFAULT_LOOPED()
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  }
  DEFAULT_LOOPED()
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  }
  DEFAULT_LOOPED()
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  }
  DEFAULT_LOOPED()
};

struct Real {
  template <typename T>
  T operator()(T x) {
    return std::real(x);
  }
  DEFAULT_LOOPED()
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return std::rint(x);
  }

  complex64_t operator()(complex64_t x) {
    return {std::rint(x.real()), std::rint(x.imag())};
  }
  DEFAULT_LOOPED()
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto one = static_cast<decltype(x)>(1.0);
    return one / (one + fast_exp(-x));
  }
  DEFAULT_LOOPED()
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  }
  uint8_t operator()(uint8_t x) {
    return x != 0;
  }
  uint16_t operator()(uint16_t x) {
    return x != 0;
  }
  uint32_t operator()(uint32_t x) {
    return x != 0;
  }
  uint64_t operator()(uint64_t x) {
    return x != 0;
  }

  complex64_t operator()(complex64_t x) {
    return x == complex64_t(0) ? x : x / std::abs(x);
  }
  DEFAULT_LOOPED()
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return std::sin(x);
  }
  DEFAULT_LOOPED()
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return std::sinh(x);
  }
  DEFAULT_LOOPED()
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  }
  DEFAULT_LOOPED()
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return std::sqrt(x);
  }
  DEFAULT_LOOPED()
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return static_cast<decltype(x)>(1.0) / std::sqrt(x);
  }
  DEFAULT_LOOPED()
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return std::tan(x);
  }
  DEFAULT_LOOPED()
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return std::tanh(x);
  }
  DEFAULT_LOOPED()
};

} // namespace mlx::core::detail
