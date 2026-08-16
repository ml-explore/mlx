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
      T r = cuda::std::fmod(x, y);
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
    using cuda::std::isnan;
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
      if (cuda::std::isnan(x.real()) || cuda::std::isnan(x.imag()) ||
          cuda::std::isnan(y.real()) || cuda::std::isnan(y.imag())) {
        return {
            cuda::std::numeric_limits<float>::quiet_NaN(),
            cuda::std::numeric_limits<float>::quiet_NaN()};
      }
      auto max = x.real() > y.real() ? x : y;
      auto min = x.real() < y.real() ? x : y;
      auto min_real = min.real();
      auto max_real = max.real();
      if (!cuda::std::isfinite(min_real) && (min_real == max_real)) {
        if (min_real < 0) {
          return min;
        } else {
          return Log{}(Exp{}(min) + Exp{}(max));
        }
      } else {
        return Log1p{}(Exp{}(min - max)) + max;
      }
    } else {
      if (cuda::std::isnan(x) || cuda::std::isnan(y)) {
        return cuda::std::numeric_limits<T>::quiet_NaN();
      }
      T maxval = max(x, y);
      T minval = min(x, y);
      return (minval == -cuda::std::numeric_limits<T>::infinity() ||
              maxval == cuda::std::numeric_limits<T>::infinity())
          ? maxval
          : T(maxval + cuda::std::log1p(cuda::std::exp(minval - maxval)));
    }
  };
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return max(x, y);
    } else if constexpr (is_complex_v<T>) {
      if (cuda::std::isnan(x.real()) || cuda::std::isnan(x.imag())) {
        return x;
      }
      return x > y ? x : y;
    } else {
      if (cuda::std::isnan(x)) {
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
      if (cuda::std::isnan(x.real()) || cuda::std::isnan(x.imag())) {
        return x;
      }
      return x < y ? x : y;
    } else {
      if (cuda::std::isnan(x)) {
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
      return cuda::std::pow(base, exp);
    } else {
      return cuda::std::pow(base, exp);
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
    return cuda::std::atan2(y, x);
  }
};

struct DivMod {
  template <typename T>
  __device__ cuda::std::array<T, 2> operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      // Integer floor-divmod without overflow: start from the truncating
      // quotient/remainder and shift both down by one when the signs differ.
      // This avoids computing (a - r), which can overflow signed integers at
      // the extremes (e.g. INT_MAX / -2).
      auto q = x / y;
      auto r = x % y;
      if (r != 0 && ((r < 0) != (y < 0))) {
        q -= 1;
        r += y;
      }
      return {q, r};
    } else if constexpr (is_complex_v<T>) {
      auto r = Remainder{}(x, y);
      return {(x - r) / y, r};
    } else {
      // numpy semantics: the quotient is floor(a / b) and the remainder
      // carries the divisor's sign, so q * b + r == a holds for every sign
      // combination. b == 0 yields a / b; nan inputs and an infinite dividend
      // yield nan, matching numpy's floor_divide. floor(a / b) matches numpy
      // bit for bit (deriving the quotient from the remainder can differ by
      // one ulp).
      auto r = cuda::std::fmod(x, y);
      T q;
      if (y == 0) {
        q = x / y;
      } else if (cuda::std::isnan(x) || cuda::std::isnan(y) ||
                 cuda::std::isinf(x)) {
        q = cuda::std::numeric_limits<T>::quiet_NaN();
      } else {
        q = cuda::std::floor(x / y);
      }
      if (r != 0 && ((r < 0) != (y < 0))) {
        r += y;
      }
      return {q, r};
    }
  };
};

} // namespace mlx::core::cu
