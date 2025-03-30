// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/erf.h"
#include "mlx/backend/metal/kernels/expm1f.h"

namespace {
constant float inf = metal::numeric_limits<float>::infinity();
}

struct Abs {
  template <typename T>
  T operator()(T x) {
    return metal::abs(x);
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
  template <>
  complex64_t operator()(complex64_t x) {
    return {metal::precise::sqrt(x.real * x.real + x.imag * x.imag), 0};
  };
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acos(x);
  };
};

struct ArcCosh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acosh(x);
  };
};

struct ArcSin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::asin(x);
  };
};

struct ArcSinh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::asinh(x);
  };
};

struct ArcTan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::atan(x);
  };
};

struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::atanh(x);
  };
};

struct BitwiseInvert {
  template <typename T>
  T operator()(T x) {
    return ~x;
  };
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return metal::ceil(x);
  };
  template <>
  int8_t operator()(int8_t x) {
    return x;
  };
  template <>
  int16_t operator()(int16_t x) {
    return x;
  };
  template <>
  int32_t operator()(int32_t x) {
    return x;
  };
  template <>
  int64_t operator()(int64_t x) {
    return x;
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cos(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::cos(x.real) * metal::precise::cosh(x.imag),
        -metal::precise::sin(x.real) * metal::precise::sinh(x.imag)};
  };
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cosh(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::cosh(x.real) * metal::precise::cos(x.imag),
        metal::precise::sinh(x.real) * metal::precise::sin(x.imag)};
  };
};

struct Conjugate {
  complex64_t operator()(complex64_t x) {
    return complex64_t{x.real, -x.imag};
  }
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erf(static_cast<float>(x)));
  };
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(erfinv(static_cast<float>(x)));
  };
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return metal::precise::exp(x);
  };
  template <>
  complex64_t operator()(complex64_t x) {
    auto m = metal::precise::exp(x.real);
    return {m * metal::precise::cos(x.imag), m * metal::precise::sin(x.imag)};
  }
};

struct Expm1 {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(expm1f(static_cast<float>(x)));
  };
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return metal::floor(x);
  };
  template <>
  int8_t operator()(int8_t x) {
    return x;
  };
  template <>
  int16_t operator()(int16_t x) {
    return x;
  };
  template <>
  int32_t operator()(int32_t x) {
    return x;
  };
  template <>
  int64_t operator()(int64_t x) {
    return x;
  };
  template <>
  uint8_t operator()(uint8_t x) {
    return x;
  };
  template <>
  uint16_t operator()(uint16_t x) {
    return x;
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x;
  };
  template <>
  uint64_t operator()(uint64_t x) {
    return x;
  };
  template <>
  bool operator()(bool x) {
    return x;
  };
};

struct Imag {
  template <typename T>
  T operator()(T x) {
    return x.imag;
  };
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    auto r = metal::precise::log(Abs{}(x).real);
    auto i = metal::precise::atan2(x.imag, x.real);
    return {r, i};
  };
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log2(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    auto y = Log{}(x);
    return {y.real / M_LN2_F, y.imag / M_LN2_F};
  };
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return metal::precise::log10(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    auto y = Log{}(x);
    return {y.real / M_LN10_F, y.imag / M_LN10_F};
  };
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  };
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  };
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  };
};

struct Real {
  template <typename T>
  T operator()(T x) {
    return x.real;
  };
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return metal::rint(x);
  };
  template <>
  complex64_t operator()(complex64_t x) {
    return {metal::rint(x.real), metal::rint(x.imag)};
  };
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  };
  template <>
  uint32_t operator()(uint32_t x) {
    return x != 0;
  };
  template <>
  complex64_t operator()(complex64_t x) {
    if (x == complex64_t(0)) {
      return x;
    }
    return x /
        (complex64_t)metal::precise::sqrt(x.real * x.real + x.imag * x.imag);
  };
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sin(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::sin(x.real) * metal::precise::cosh(x.imag),
        metal::precise::cos(x.real) * metal::precise::sinh(x.imag)};
  };
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sinh(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    return {
        metal::precise::sinh(x.real) * metal::precise::cos(x.imag),
        metal::precise::cosh(x.real) * metal::precise::sin(x.imag)};
  };
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  };
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::sqrt(x);
  };
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::rsqrt(x);
  };
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tan(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    float tan_a = metal::precise::tan(x.real);
    float tanh_b = metal::precise::tanh(x.imag);
    float t1 = tan_a * tanh_b;
    float denom = 1. + t1 * t1;
    return {(tan_a - tanh_b * t1) / denom, (tanh_b + tan_a * t1) / denom};
  };
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tanh(x);
  };

  template <>
  complex64_t operator()(complex64_t x) {
    float tanh_a = metal::precise::tanh(x.real);
    float tan_b = metal::precise::tan(x.imag);
    float t1 = tanh_a * tan_b;
    float denom = 1. + t1 * t1;
    return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
  };
};
