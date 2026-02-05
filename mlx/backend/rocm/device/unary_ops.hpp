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
    } else if constexpr (std::is_same_v<T, float>) {
      return fabsf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return fabs(x);
    } else if constexpr (std::is_same_v<T, __half>) {
      return __habs(x);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return hip_bfloat16(fabsf(static_cast<float>(x)));
    } else if constexpr (is_complex_v<T>) {
      return make_hipFloatComplex(hypotf(x.x, x.y), 0.0f);
    } else {
      // For integral types
      return abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::acosf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::acos(x);
    } else {
      return acos(x);
    }
  }
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::acoshf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::acosh(x);
    } else {
      return acosh(x);
    }
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::asinf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::asin(x);
    } else {
      return asin(x);
    }
  }
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::asinhf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::asinh(x);
    } else {
      return asinh(x);
    }
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::atanf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::atan(x);
    } else {
      return atan(x);
    }
  }
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::atanhf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::atanh(x);
    } else {
      return atanh(x);
    }
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
      return T{::ceilf(x.x), ::ceilf(x.y)};
    } else if constexpr (std::is_same_v<T, float>) {
      return ::ceilf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::ceil(x);
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
    if constexpr (std::is_same_v<T, float>) {
      return ::coshf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::cosh(x);
    } else {
      return cosh(x);
    }
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
      return T{::floorf(x.x), ::floorf(x.y)};
    } else if constexpr (std::is_same_v<T, float>) {
      return ::floorf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::floor(x);
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
    } else if constexpr (std::is_same_v<T, float>) {
      return ::log2f(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::log2(x);
    } else {
      return log2(x);
    }
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::log10f(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::log10(x);
    } else {
      return log10(x);
    }
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
      return {::rintf(x.x), ::rintf(x.y)};
    } else if constexpr (std::is_same_v<T, float>) {
      return ::rintf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::rint(x);
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
    if constexpr (std::is_same_v<T, float>) {
      return ::sinhf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::sinh(x);
    } else {
      return sinh(x);
    }
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
    if constexpr (std::is_same_v<T, float>) {
      return ::sqrtf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::sqrt(x);
    } else {
      return sqrt(x);
    }
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (is_complex_v<T>) {
      return hipCdivf(make_hipFloatComplex(1.0f, 0.0f), Sqrt{}(x));
    } else if constexpr (std::is_same_v<T, float>) {
      return ::rsqrtf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::rsqrt(x);
    } else {
      return rsqrt(x);
    }
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::tanf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::tan(x);
    } else {
      return tan(x);
    }
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (std::is_same_v<T, float>) {
      return ::tanhf(x);
    } else if constexpr (std::is_same_v<T, double>) {
      return ::tanh(x);
    } else {
      return tanh(x);
    }
  }
};

} // namespace mlx::core::rocm
