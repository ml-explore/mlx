// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cuda_fp8.h>
#include <math_constants.h>
#include <cuda/std/cmath>

namespace mlx::core::cu {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x;
    } else {
      return cuda::std::abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::acos(x);
  }
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::asin(x);
  }
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::atan(x);
  }
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::atanh(x);
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
    } else if constexpr (is_complex_v<T>) {
      return T{cuda::std::ceil(x.real()), cuda::std::ceil(x.imag())};
    } else {
      return cuda::std::ceil(x);
    }
  }
};

struct Conjugate {
  template <typename T>
  __device__ complex_t<T> operator()(complex_t<T> x) {
    return cuda::std::conj(x);
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::cos(x);
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::cosh(x);
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

struct LogGamma {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return ::lgamma(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return ::lgamma(__bfloat162float(x));
    } else {
      return ::lgamma(x);
    }
  }
};

struct Digamma {
  template <typename T>
  __device__ T operator()(T x) {
    float v = static_cast<float>(x);
    float r = 0.0f;
    if (v < 0.0f) {
      r = -M_PI / ::tan(M_PI * v);
      v = 1.0f - v;
    }
    while (v < 10.0f) {
      r -= 1.0f / v;
      v += 1.0f;
    }
    float z = 1.0f / (v * v);
    float y = 3.96825396825e-3f;
    y = y * z + (-4.16666666667e-3f);
    y = y * z + 7.57575757576e-3f;
    y = y * z + (-2.10927960928e-2f);
    y = y * z + 8.33333333333e-2f;
    r += ::logf(v) - 0.5f / v - y * z;
    return static_cast<T>(r);
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::exp(x);
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::expm1(x);
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{cuda::std::floor(x.real()), cuda::std::floor(x.imag())};
    } else {
      return cuda::std::floor(x);
    }
  }
};

struct Imag {
  template <typename T>
  __device__ auto operator()(complex_t<T> x) {
    return x.imag();
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::log(x);
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      auto y = Log{}(x);
      return {y.real() / CUDART_LN2_F, y.imag() / CUDART_LN2_F};
    } else {
      return cuda::std::log2(x);
    }
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::log10(x);
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T z) {
    if constexpr (is_complex_v<T>) {
      float x = z.real();
      float y = z.imag();
      float zabs = Abs{}(z).real();
      float theta = atan2f(y, x + 1);
      if (zabs < 0.5f) {
        float r = x * (2 + x) + y * y;
        if (r == 0) { // handle underflow
          return {x, theta};
        }
        return {0.5f * log1pf(r), theta};
      } else {
        float z0 = hypotf(x + 1, y);
        return {logf(z0), theta};
      }
    } else {
      return cuda::std::log1p(z);
    }
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
    if constexpr (is_complex_v<T>) {
      return T{0, 0} - x;
    } else {
      return -x;
    }
  }
};

struct Real {
  template <typename T>
  __device__ auto operator()(complex_t<T> x) {
    return x.real();
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return {cuda::std::rint(x.real()), cuda::std::rint(x.imag())};
    } else {
      return cuda::std::rint(x);
    }
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T y = 1 / (1 + cuda::std::exp(cuda::std::abs(x)));
    return (x < 0) ? y : 1 - y;
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x != 0;
    } else if constexpr (is_complex_v<T>) {
      if (x.real() == 0 && x.imag() == 0) {
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
    return cuda::std::sin(x);
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::sinh(x);
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
    return cuda::std::sqrt(x);
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return 1.0f / Sqrt{}(x);
    } else if constexpr (cuda::std::is_same_v<T, __half>) {
      return rsqrt(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return rsqrt(__bfloat162float(x));
    } else {
      return rsqrt(x);
    }
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::tan(x);
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    return cuda::std::tanh(x);
  }
};

struct ToFP8 {
  template <typename T>
  __device__ uint8_t operator()(T x) {
    return __nv_fp8_e4m3(x).__x;
  }
};

struct FromFP8 {
  __device__ float operator()(uint8_t x) {
    return float(*(__nv_fp8_e4m3*)(&x));
  }
};

} // namespace mlx::core::cu
