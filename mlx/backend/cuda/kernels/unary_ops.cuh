// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/kernels/fp16_math.cuh"
#include "mlx/backend/cuda/kernels/utils.cuh"

namespace mlx::core::cu {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x;
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {sqrt(cuCrealf(x) * cuCrealf(x) + cuCimagf(x) * cuCimagf(x)), 0};
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
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else {
      return ceil(x);
    }
  }
};

struct Conjugate {
  __device__ cuComplex operator()(cuComplex x) {
    return {cuCrealf(x), -cuCimagf(x)};
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          cos(cuCrealf(x)) * cosh(cuCimagf(x)),
          -sin(cuCrealf(x)) * sinh(cuCimagf(x))};
    } else {
      return cos(x);
    }
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          cosh(cuCrealf(x)) * cos(cuCimagf(x)),
          sinh(cuCrealf(x)) * sin(cuCimagf(x))};
    } else {
      return cosh(x);
    }
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return erf(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return erf(__bfloat162float(x));
    } else {
      return erf(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return erfinv(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return erfinv(__bfloat162float(x));
    } else {
      return erfinv(x);
    }
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      auto m = exp(cuCrealf(x));
      return {m * cos(cuCimagf(x)), m * sinh(cuCimagf(x))};
    } else {
      return exp(x);
    }
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return expm1(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return expm1(__bfloat162float(x));
    } else {
      return expm1(x);
    }
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else {
      return floor(x);
    }
  }
};

struct Imag {
  __device__ float operator()(cuComplex x) {
    return cuCimagf(x);
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    return log(x);
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    return log2(x);
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    return log10(x);
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
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return 0 - x;
    } else {
      return -x;
    }
  }
};

struct Real {
  __device__ float operator()(cuComplex x) {
    return cuCrealf(x);
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {rint(cuCrealf(x)), rint(cuCimagf(x))};
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
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x != 0;
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      if (cuCrealf(x) == 0 && cuCimagf(x) == 0) {
        return x;
      } else {
        return x / Abs()(x);
      }
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return static_cast<float>((x > T(0.f)) - (x < T(0.f)));
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          sin(cuCrealf(x)) * cosh(cuCimagf(x)),
          cos(cuCrealf(x)) * sinh(cuCimagf(x))};
    } else {
      return sin(x);
    }
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          sinh(cuCrealf(x)) * cos(cuCimagf(x)),
          cosh(cuCrealf(x)) * sin(cuCimagf(x))};
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
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      float tan_a = tan(cuCrealf(x));
      float tanh_b = tanh(cuCimagf(x));
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
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      float tanh_a = tanh(cuCrealf(x));
      float tan_b = tan(cuCimagf(x));
      float t1 = tanh_a * tan_b;
      float denom = 1. + t1 * t1;
      return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
    } else {
      return tanh(x);
    }
  }
};

} // namespace mlx::core::cu
