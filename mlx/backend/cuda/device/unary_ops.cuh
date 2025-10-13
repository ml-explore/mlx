// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <math_constants.h>

namespace mlx::core::cu {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
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
    return ~x;
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else if constexpr (is_complex_v<T>) {
      return T{ceil(x.real()), ceil(x.imag())};
    } else {
      return ceil(x);
    }
  }
};

struct Conjugate {
  template <typename T>
  __device__ complex_t<T> operator()(complex_t<T> x) {
    return conj(x);
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    return cos(x);
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
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return erf(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return erf(__bfloat162float(x));
    } else {
      return erf(x);
    }
  

… [truncated 2298 chars] …

or()(T x) {
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
      return {rint(x.real()), rint(x.imag())};
    } else {
      return rint(x);
    }
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T one = T(1);
    T y = one / (one + exp(abs(x)));
    return (x < 0) ? y : one - y;
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
    return sin(x);
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
    return x * x;
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
      return 1.0f / Sqrt{}(x);
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

} // namespace mlx::core::cu