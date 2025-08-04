// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/cexpf.h"
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
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
  complex64_t operator()(complex64_t x) {
    return {metal::precise::sqrt(x.real * x.real + x.imag * x.imag), 0};
  };
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::acos(x);
  };

  complex64_t operator()(complex64_t x);
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

  complex64_t operator()(complex64_t x);
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

  complex64_t operator()(complex64_t x);
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
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return metal::precise::cos(x);
  };

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
  complex64_t operator()(complex64_t x) {
    return cexpf(x);
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
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
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
  uint32_t operator()(uint32_t x) {
    return x != 0;
  };
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

  complex64_t operator()(complex64_t x) {
    if (x.real == 0.0 && x.imag == 0.0) {
      return {0.0, 0.0};
    }
    auto r = Abs{}(x).real;
    auto a = metal::precise::sqrt((r + x.real) / 2.0);
    auto b_abs = metal::precise::sqrt((r - x.real) / 2.0);
    auto b = metal::copysign(b_abs, x.imag);
    return {a, b};
  }
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return metal::precise::rsqrt(x);
  };

  complex64_t operator()(complex64_t x) {
    return 1.0 / Sqrt{}(x);
  }
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return metal::precise::tan(x);
  };

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

  complex64_t operator()(complex64_t x) {
    float tanh_a = metal::precise::tanh(x.real);
    float tan_b = metal::precise::tan(x.imag);
    float t1 = tanh_a * tan_b;
    float denom = 1. + t1 * t1;
    return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
  };
};

complex64_t ArcCos::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto y = Log{}(x + i * Sqrt{}(1.0 - x * x));
  return {y.imag, -y.real};
};

complex64_t ArcSin::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto y = Log{}(i * x + Sqrt{}(1.0 - x * x));
  return {y.imag, -y.real};
};

complex64_t ArcTan::operator()(complex64_t x) {
  auto i = complex64_t{0.0, 1.0};
  auto ix = i * x;
  return (1.0 / complex64_t{0.0, 2.0}) * Log{}((1.0 + ix) / (1.0 - ix));
};
