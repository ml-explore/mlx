// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/unary_ops.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

struct Add {
  template <typename T>
  __device__ T operator()(T x, T y) {
    return x + y;
  }
};

struct FloorDivide {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return x / y;
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return hip_bfloat16(
          truncf(static_cast<float>(x) / static_cast<float>(y)));
    } else if constexpr (std::is_same_v<T, __half>) {
      return __float2half(truncf(__half2float(x) / __half2float(y)));
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
    if constexpr (std::is_integral_v<T>) {
      if constexpr (std::is_signed_v<T>) {
        auto r = x % y;
        if (r != 0 && (r < 0 != y < 0)) {
          r += y;
        }
        return r;
      } else {
        return x % y;
      }
    } else if constexpr (is_complex_v<T>) {
      // Complex modulo not typically defined, return x
      return x;
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      float fx = static_cast<float>(x);
      float fy = static_cast<float>(y);
      float r = fmodf(fx, fy);
      if (r != 0 && (r < 0 != fy < 0)) {
        r = r + fy;
      }
      return hip_bfloat16(r);
    } else if constexpr (std::is_same_v<T, __half>) {
      float fx = __half2float(x);
      float fy = __half2float(y);
      float r = fmodf(fx, fy);
      if (r != 0 && (r < 0 != fy < 0)) {
        r = r + fy;
      }
      return __float2half(r);
    } else {
      T r = fmodf(x, y);
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
      return (x.x == y.x && x.y == y.y) ||
          (__isnanf(x.x) && __isnanf(y.x) && __isnanf(x.y) && __isnanf(y.y)) ||
          (x.x == y.x && __isnanf(x.y) && __isnanf(y.y)) ||
          (__isnanf(x.x) && __isnanf(y.x) && x.y == y.y);
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      float fx = static_cast<float>(x);
      float fy = static_cast<float>(y);
      return fx == fy || (__isnanf(fx) && __isnanf(fy));
    } else if constexpr (std::is_same_v<T, __half>) {
      float fx = __half2float(x);
      float fy = __half2float(y);
      return fx == fy || (__isnanf(fx) && __isnanf(fy));
    } else {
      return x == y || (__isnanf(x) && __isnanf(y));
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
    if constexpr (std::is_integral_v<T>) {
      // LogAddExp doesn't make sense for integers, but handle it gracefully
      return x > y ? x : y;
    } else if constexpr (is_complex_v<T>) {
      if (isnan(x.x) || isnan(x.y) || isnan(y.x) || isnan(y.y)) {
        return {
            numeric_limits<float>::quiet_NaN(),
            numeric_limits<float>::quiet_NaN()};
      }
      auto maxv = x.x > y.x ? x : y;
      auto minv = x.x < y.x ? x : y;
      auto min_real = minv.x;
      auto max_real = maxv.x;
      if (!isfinite(min_real) && (min_real == max_real)) {
        if (min_real < 0) {
          return minv;
        } else {
          return Log{}(hipCaddf(Exp{}(minv), Exp{}(maxv)));
        }
      } else {
        return hipCaddf(Log1p{}(Exp{}(hipCsubf(minv, maxv))), maxv);
      }
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      float fx = static_cast<float>(x);
      float fy = static_cast<float>(y);
      if (isnan(fx) || isnan(fy)) {
        return hip_bfloat16(numeric_limits<float>::quiet_NaN());
      }
      float maxval = fmaxf(fx, fy);
      float minval = fminf(fx, fy);
      float result = (minval == -numeric_limits<float>::infinity() ||
                      maxval == numeric_limits<float>::infinity())
          ? maxval
          : maxval + log1pf(expf(minval - maxval));
      return hip_bfloat16(result);
    } else if constexpr (std::is_same_v<T, __half>) {
      float fx = __half2float(x);
      float fy = __half2float(y);
      if (isnan(fx) || isnan(fy)) {
        return __float2half(numeric_limits<float>::quiet_NaN());
      }
      float maxval = fmaxf(fx, fy);
      float minval = fminf(fx, fy);
      float result = (minval == -numeric_limits<float>::infinity() ||
                      maxval == numeric_limits<float>::infinity())
          ? maxval
          : maxval + log1pf(expf(minval - maxval));
      return __float2half(result);
    } else {
      if (isnan(x) || isnan(y)) {
        return numeric_limits<T>::quiet_NaN();
      }
      T maxval = fmaxf(x, y);
      T minval = fminf(x, y);
      return (minval == -numeric_limits<T>::infinity() ||
              maxval == numeric_limits<T>::infinity())
          ? maxval
          : T(float(maxval) + log1pf(expf(minval - maxval)));
    }
  };
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return max(x, y);
    } else if constexpr (is_complex_v<T>) {
      if (__isnanf(x.x) || __isnanf(x.y)) {
        return x;
      }
      // Compare by real part first, then imaginary
      if (x.x > y.x || (x.x == y.x && x.y > y.y)) {
        return x;
      }
      return y;
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      float fx = static_cast<float>(x);
      float fy = static_cast<float>(y);
      if (__isnanf(fx)) {
        return x;
      }
      return fx > fy ? x : y;
    } else if constexpr (std::is_same_v<T, __half>) {
      float fx = __half2float(x);
      float fy = __half2float(y);
      if (__isnanf(fx)) {
        return x;
      }
      return fx > fy ? x : y;
    } else {
      if (__isnanf(x)) {
        return x;
      }
      return x > y ? x : y;
    }
  }
};

struct Minimum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return min(x, y);
    } else if constexpr (is_complex_v<T>) {
      if (__isnanf(x.x) || __isnanf(x.y)) {
        return x;
      }
      // Compare by real part first, then imaginary
      if (x.x < y.x || (x.x == y.x && x.y < y.y)) {
        return x;
      }
      return y;
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      float fx = static_cast<float>(x);
      float fy = static_cast<float>(y);
      if (__isnanf(fx)) {
        return x;
      }
      return fx < fy ? x : y;
    } else if constexpr (std::is_same_v<T, __half>) {
      float fx = __half2float(x);
      float fy = __half2float(y);
      if (__isnanf(fx)) {
        return x;
      }
      return fx < fy ? x : y;
    } else {
      if (__isnanf(x)) {
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
      return x.x != y.x || x.y != y.y;
    } else {
      return x != y;
    }
  }
};

struct Power {
  template <typename T>
  __device__ T operator()(T base, T exp) {
    if constexpr (std::is_integral_v<T>) {
      T res = 1;
      // Raising an integer to a negative power is undefined
      if constexpr (std::is_signed_v<T>) {
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
      // Complex power: base^exp = exp(exp * log(base))
      float r = hypotf(base.x, base.y);
      float theta = atan2f(base.y, base.x);
      float log_r = logf(r);
      float new_r = expf(exp.x * log_r - exp.y * theta);
      float new_theta = exp.x * theta + exp.y * log_r;
      return make_hipFloatComplex(
          new_r * cosf(new_theta), new_r * sinf(new_theta));
    } else if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return hip_bfloat16(
          powf(static_cast<float>(base), static_cast<float>(exp)));
    } else if constexpr (std::is_same_v<T, __half>) {
      return __float2half(powf(__half2float(base), __half2float(exp)));
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
  __device__ bool operator()(T x, T y) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return (static_cast<float>(x) != 0.0f) && (static_cast<float>(y) != 0.0f);
    } else if constexpr (std::is_same_v<T, __half>) {
      return (__half2float(x) != 0.0f) && (__half2float(y) != 0.0f);
    } else if constexpr (std::is_floating_point_v<T>) {
      return (x != T(0)) && (y != T(0));
    } else {
      return x && y;
    }
  };
};

struct LogicalOr {
  template <typename T>
  __device__ bool operator()(T x, T y) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return (static_cast<float>(x) != 0.0f) || (static_cast<float>(y) != 0.0f);
    } else if constexpr (std::is_same_v<T, __half>) {
      return (__half2float(x) != 0.0f) || (__half2float(y) != 0.0f);
    } else if constexpr (std::is_floating_point_v<T>) {
      return (x != T(0)) || (y != T(0));
    } else {
      return x || y;
    }
  };
};

struct BitwiseAnd {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return x & y;
    } else {
      // This branch should never be taken due to supports_binary_op filtering
      return T{};
    }
  };
};

struct BitwiseOr {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return x | y;
    } else {
      return T{};
    }
  };
};

struct BitwiseXor {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return x ^ y;
    } else {
      return T{};
    }
  };
};

struct LeftShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return x << y;
    } else {
      return T{};
    }
  };
};

struct RightShift {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      return x >> y;
    } else {
      return T{};
    }
  };
};

struct ArcTan2 {
  template <typename T>
  __device__ T operator()(T y, T x) {
    if constexpr (std::is_same_v<T, hip_bfloat16>) {
      return hip_bfloat16(atan2f(static_cast<float>(y), static_cast<float>(x)));
    } else if constexpr (std::is_same_v<T, __half>) {
      return __float2half(atan2f(__half2float(y), __half2float(x)));
    } else if constexpr (std::is_same_v<T, double>) {
      return atan2(y, x);
    } else {
      return atan2f(y, x);
    }
  }
};

struct DivMod {
  template <typename T>
  __device__ hip_array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};

} // namespace mlx::core::rocm
