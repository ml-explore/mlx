// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/fp16_math.hpp"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned_v<T>) {
      return x;
    } else if constexpr (std::is_same_v<T, hipFloatComplex>) {
      return {
          sqrt(hipCrealf(x) * hipCrealf(x) + hipCimagf(x) * hipCimagf(x)), 0};
    } else {
      return abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    return acos(x);
  }
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    return acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    return asin(x);
  }
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    return asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    return atan(x);
  }
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    return atanh(x);
  }
};

struct BitwiseInvert {
  template <typename T>
  __device__ T operator()(T x) {
    return ~x;
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else {
      return ceil(x);
    }
  }
};

struct Conjugate {
  __device__ hipFloatComplex operator()(hipFloatComplex x) {
    return {hipCrealf(x), -hipCimagf(x)};
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      return {
          cos(hipCrealf(x)) * cosh(hipCimagf(x)),
          -sin(hipCrealf(x)) * sinh(hipCimagf(x))};
    } else {
      return cos(x);
    }
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      return {
          cosh(hipCrealf(x)) * cos(hipCimagf(x)),
          sinh(hipCrealf(x)) * sin(hipCimagf(x))};
    } else {
      return cosh(x);
    }
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return erf(__half2float(x));
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return erf(__bfloat162float(x));
    } else {
      return erf(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return erfinv(__half2float(x));
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return erfinv(__bfloat162float(x));
    } else {
      return erfinv(x);
    }
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      auto m = exp(hipCrealf(x));
      return {m * cos(hipCimagf(x)), m * sinh(hipCimagf(x))};
    } else {
      return exp(x);
    }
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, __half>) {
      return expm1(__half2float(x));
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return expm1(__bfloat162float(x));
    } else {
      return expm1(x);
    }
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else {
      return floor(x);
    }
  }
};

struct Imag {
  __device__ float operator()(hipFloatComplex x) {
    return hipCimagf(x);
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      auto r = log(hipCrealf(Abs{}(x)));
      auto i = atan2f(hipCimagf(x), hipCrealf(x));
      return {r, i};
    } else {
      return log(x);
    }
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      auto y = Log{}(x);
      return {hipCrealf(y) / M_LN2, hipCimagf(y) / M_LN2};
    } else {
      return log2(x);
    }
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      auto y = Log{}(x);
      return {hipCrealf(y) / M_LN10, hipCimagf(y) / M_LN10};
    } else {
      return log10(x);
    }
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T x) {
    return log1p(x);
  }
};

struct LogicalNot {
  __device__ bool operator()(bool x) {
    return !x;
  }
};

struct Negative {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      return 0 - x;
    } else {
      return -x;
    }
  }
};

struct Real {
  __device__ float operator()(hipFloatComplex x) {
    return hipCrealf(x);
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      return {rint(hipCrealf(x)), rint(hipCimagf(x))};
    } else {
      return rint(x);
    }
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return rsqrt(x);
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T y = 1 / (1 + exp(-abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned_v<T>) {
      return x != 0;
    } else if constexpr (std::is_same_v<T, hipFloatComplex>) {
      if (hipCrealf(x) == 0 && hipCimagf(x) == 0) {
        return x;
      } else {
        return x / Abs()(x);
      }
    } else if constexpr (std::is_same_v<T, __hip_bfloat16>) {
      return static_cast<float>((x > T(0.f)) - (x < T(0.f)));
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      return {
          sin(hipCrealf(x)) * cosh(hipCimagf(x)),
          cos(hipCrealf(x)) * sinh(hipCimagf(x))};
    } else {
      return sin(x);
    }
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      return {
          sinh(hipCrealf(x)) * cos(hipCimagf(x)),
          cosh(hipCrealf(x)) * sin(hipCimagf(x))};
    } else {
      return sinh(x);
    }
  }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    return x * x;
  }
};

struct Sqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return sqrt(x);
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      float tan_a = tan(hipCrealf(x));
      float tanh_b = tanh(hipCimagf(x));
      float t1 = tan_a * tanh_b;
      float denom = 1. + t1 * t1;
      return {(tan_a - tanh_b * t1) / denom, (tanh_b + tan_a * t1) / denom};
    } else {
      return tan(x);
    }
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hipFloatComplex>) {
      float tanh_a = tanh(hipCrealf(x));
      float tan_b = tan(hipCimagf(x));
      float t1 = tanh_a * tan_b;
      float denom = 1. + t1 * t1;
      return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
    } else {
      return tanh(x);
    }
  }
};

} // namespace mlx::core::rocm