// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/kernels/fp16_math.cuh"

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
      return trunc(x / y);
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
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
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
    if constexpr (std::is_same_v<T, cuComplex>) {
      return x == y ||
          (isnan(cuCrealf(x)) && isnan(cuCrealf(y)) && isnan(cuCimagf(x)) &&
           isnan(cuCimagf(y))) ||
          (cuCrealf(x) == cuCrealf(y) && isnan(cuCimagf(x)) &&
           isnan(cuCimagf(y))) ||
          (isnan(cuCrealf(x)) && isnan(cuCrealf(y)) &&
           cuCimagf(x) == cuCimagf(y));
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
    if (isnan(x) || isnan(y)) {
      return cuda::std::numeric_limits<T>::quiet_NaN();
    }
    T maxval = max(x, y);
    T minval = min(x, y);
    return (minval == -cuda::std::numeric_limits<T>::infinity() ||
            maxval == cuda::std::numeric_limits<T>::infinity())
        ? maxval
        : T(float(maxval) + log1p(expf(minval - maxval)));
  };
};

struct Maximum {
  template <typename T>
  __device__ T operator()(T x, T y) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return max(x, y);
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      if (isnan(cuCrealf(x)) || isnan(cuCimagf(x))) {
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
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      if (isnan(cuCrealf(x)) || isnan(cuCimagf(x))) {
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
    if constexpr (std::is_same_v<T, cuComplex>) {
      return cuCrealf(x) != cuCrealf(y) || cuCimagf(x) != cuCimagf(y);
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
      while (exp) {
        if (exp & 1) {
          res *= base;
        }
        exp >>= 1;
        base *= base;
      }
      return res;
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      auto x_theta = atan2f(base.y, base.x);
      auto x_ln_r = 0.5 * logf(base.x * base.x + base.y * base.y);
      auto mag = expf(exp.x * x_ln_r - exp.y * x_theta);
      auto phase = exp.y * x_ln_r + exp.x * x_theta;
      return make_cuFloatComplex(mag * cosf(phase), mag * sinf(phase));
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
