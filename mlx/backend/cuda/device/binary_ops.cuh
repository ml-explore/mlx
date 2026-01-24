// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device/unary_ops.cuh"

#include <cuda/std/array>

namespace mlx::core::cu {

struct Add {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x + y;
  }
};

struct FloorDivide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x / y;
    } else {
      return truncf(x / y);
    }
  }
};

struct Divide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      if constexpr (cuda::std::is_signed_v<T>) {
        auto r = x % y;
        if (r != 0 && (r < 0 != y < 0)) {
          r += y;
        }
        return r;
      } else {
        return x % y;
      }
    } else if constexpr (is_complex_v<T>) {
      return x % y;
    } else {
      T r = fmod(x, y);
      if (r != 0 && (r < 0 != y < 0)) {
        r = r + y;
      }
      return r;
    }
  }
};

struct Equal {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (is_complex_v<T>) {
      return x == y ||
          (isnan(x.real()) && isnan(y.real()) && isnan(x.imag()) &&
           isnan(y.imag())) ||
          (x.real() == y.real() && isnan(x.imag()) && isnan(y.imag())) ||
          (isnan(x.real()) && isnan(y.real()) && x.imag() == y.imag());
    } else {
      return x == y || (isnan(x) && isnan(y));
    }
  }
};

struct Greater {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    return x <= y;
  }
};

struct LogAddExp {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (is_complex_v<T>) {
      if (isnan(x.real()) || isnan(x.imag()) || isnan(y.real()) ||
          isnan(y.imag())) {
        return {
            cuda::std::numeric_limits<float>::quiet_NaN(),
            cuda::std::numeric_limits<float>::quiet_NaN()};
      }
      auto max = x.real() > y.real() ? x : y;
      auto min = x.real() < y.real() ? x : y;
      auto min_real = min.real();
      auto max_real = max.real();
      if (!isfinite(min_real) && (min_real == max_real)) {
        if (min_real < 0) {
          return min;
        } else {
          return Log{}(Exp{}(min) + Exp{}(max));
        }
      } else {
        return Log1p{}(Exp{}(min - max)) + max;
      }
    } else {
      if (isnan(x) || isnan(y)) {
        return cuda::std::numeric_limits<T>::quiet_NaN();
      }
      T maxval = max(x, y);
      T minval = min(x, y);
      return (minval == -cuda::std::numeric_limits<T>::infinity() ||
              maxval == cuda::std::numeric_limits<T>::infinity())
          ? maxval
          : T(float(maxval) + log1p(expf(minval - maxval)));
    }
  };
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return max(x, y);
    } else if constexpr (is_complex_v<T>) {
      if (isnan(x.real()) || isnan(x.imag())) {
        return x;
      }
      return x > y ? x : y;
    } else {
      if (isnan(x)) {
        return x;
      }
      return x > y ? x : y;
    }
  }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return min(x, y);
    } else if constexpr (is_complex_v<T>) {
      if (isnan(x.real()) || isnan(x.imag())) {
        return x;
      }
      return x < y ? x : y;
    } else {
      if (isnan(x)) {
        return x;
      }
      return x < y ? x : y;
    }
  }
};

struct Multiply {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (is_complex_v<T>) {
      return x.real() != y.real() || x.imag() != y.imag();
    } else {
      return x != y;
    }
  }
};

struct Power {
  template <typename T>
  __device__ T operator()(T base, T exp) {
    if constexpr (cuda::std::is_integral_v<T>) {
      T res = 1;
      // Raising an integer to a negative power is undefined
      if constexpr (cuda::std::is_signed_v<T>) {
        if (exp < 0) {
          return 0;
        }
      }
      while (exp) {
        if (exp & 1) {
          res *= base;
        }
        exp >>= 1;
        base *= base;
      }
      return res;
    } else if constexpr (is_complex_v<T>) {
      return pow(base, exp);
    } else {
      return powf(base, exp);
    }
  }
};

struct Subtract {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x && y;
  };
};

struct LogicalOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x || y;
  };
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x & y;
  };
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x | y;
  };
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x ^ y;
  };
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x << y;
  };
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x >> y;
  };
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T y, T x) {
    return atan2f(y, x);
  }
};

struct DivMod {
  template <typename T>
  __device__ cuda::std::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};

} // namespace mlx::core::cu
