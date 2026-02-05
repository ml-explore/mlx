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
    if constexpr (std::is_integral_v<T>) {
      return ~x;
    } else {
      // BitwiseInvert only makes sense for integral types
      return T{};
    }
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{ceil(x.x), ceil(x.y)};
    } else {
      return ceil(x);
    }
  }
};

struct Conjugate {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return hipConjf(x);
    } else {
      // For non-complex types, conjugate is identity
      return x;
    }
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return cosf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::cos(x);
    } else {
      return cos(x);
    }
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    return cosh(x);
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, bool> || std::is_integral_v<T>) {
      return static_cast<T>(erff(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, __half>) {
      return erf(x);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return erf(x);
    } else {
      return erff(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, bool> || std::is_integral_v<T>) {
      return static_cast<T>(erfinvf(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, __half>) {
      return erfinv(x);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return erfinv(x);
    } else {
      return erfinvf(x);
    }
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return expf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::exp(x);
    } else {
      return exp(x);
    }
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, bool> || std::is_integral_v<T>) {
      return static_cast<T>(expm1f(static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, __half>) {
      return expm1(x);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return expm1(x);
    } else {
      return expm1f(x);
    }
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{floor(x.x), floor(x.y)};
    } else {
      return floor(x);
    }
  }
};

struct Imag {
  template <typename T>
  __device__ auto operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return x.y;
    } else {
      // For non-complex types, imaginary part is 0
      return T(0);
    }
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return logf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::log(x);
    } else {
      return log(x);
    }
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      auto y = Log{}(x);
      constexpr float ln2 = 0.693147180559945309417232121458176568f;
      return {y.x / ln2, y.y / ln2};
    } else {
      return log2(x);
    }
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
  __device__ T operator()(T z) {
    if constexpr (is_complex_v<T>) {
      float x = z.x;
      float y = z.y;
      float zabs = Abs{}(z).x;
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
    } else if constexpr (std::is_same_v<T, float>) {
      return log1pf(z);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::log1p(z);
    } else {
      return log1p(z);
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
      return make_hipFloatComplex(-x.x, -x.y);
    } else {
      return -x;
    }
  }
};

struct Real {
  template <typename T>
  __device__ auto operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return x.x;
    } else {
      // For non-complex types, real part is the value itself
      return x;
    }
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return {rint(x.x), rint(x.y)};
    } else {
      return rint(x);
    }
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
      float fx = static_cast<float>(x);
      float y = 1.0f / (1.0f + expf(-fabsf(fx)));
      return T((fx < 0.0f) ? 1.0f - y : y);
    } else if constexpr (std::is_same_v<T, __half>) {
      float fx = __half2float(x);
      float y = 1.0f / (1.0f + expf(-fabsf(fx)));
      return __float2half((fx < 0.0f) ? 1.0f - y : y);
    } else {
      float fx = static_cast<float>(x);
      float y = 1.0f / (1.0f + expf(-fabsf(fx)));
      return T((fx < 0.0f) ? 1.0f - y : y);
    }
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_unsigned_v<T>) {
      return x != 0;
    } else if constexpr (is_complex_v<T>) {
      if (x.x == 0 && x.y == 0) {
        return x;
      } else {
        return hipCdivf(x, Abs()(x));
      }
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      float fx = static_cast<float>(x);
      return T((fx > 0.0f) - (fx < 0.0f));
    } else if constexpr (std::is_same_v<T, __half>) {
      float fx = __half2float(x);
      return __float2half((fx > 0.0f) - (fx < 0.0f));
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return sinf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::sin(x);
    } else {
      return sin(x);
    }
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    return sinh(x);
  }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return hipCmulf(x, x);
    } else {
      return x * x;
    }
  }
};

struct Sqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return sqrt(x);
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return hipCdivf(make_hipFloatComplex(1.0f, 0.0f), Sqrt{}(x));
    } else {
      return rsqrt(x);
    }
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    return tan(x);
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    return tanh(x);
  }
};

} // namespace mlx::core::rocm
