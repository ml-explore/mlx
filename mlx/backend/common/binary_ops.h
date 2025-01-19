// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <stdint.h>
#include <cmath>
#include <complex>

#include "mlx/backend/common/math.h"

namespace mlx::core::detail {

struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
  }
};

struct ArcTan2 {
  template <typename T>
  T operator()(T y, T x) {
    return std::atan2(y, x);
  }
};

struct Divide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
};

struct Remainder {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T> & !std::is_signed_v<T>, T> operator()(
      T numerator,
      T denominator) {
    return numerator % denominator;
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T> & std::is_signed_v<T>, T> operator()(
      T numerator,
      T denominator) {
    auto r = numerator % denominator;
    if (r != 0 && (r < 0 != denominator < 0))
      r += denominator;
    return r;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(
      T numerator,
      T denominator) {
    auto r = std::fmod(numerator, denominator);
    if (r != 0 && (r < 0 != denominator < 0)) {
      r += denominator;
    }
    return r;
  }

  complex64_t operator()(complex64_t numerator, complex64_t denominator) {
    return numerator % denominator;
  }
};

struct Equal {
  template <typename T>
  bool operator()(T x, T y) {
    return x == y;
  }
};

struct NaNEqual {
  template <typename T>
  bool operator()(T x, T y) {
    if constexpr (std::is_integral_v<T>) {
      // isnan always returns false for integers, and MSVC refuses to compile.
      return x == y;
    } else {
      return x == y || (std::isnan(x) && std::isnan(y));
    }
  }
};

struct Greater {
  template <typename T>
  bool operator()(T x, T y) {
    return x > y;
  }
};

struct GreaterEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x >= y;
  }
};

struct Less {
  template <typename T>
  bool operator()(T x, T y) {
    return x < y;
  }
};

struct LessEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x <= y;
  }
};

struct Maximum {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T x, T y) {
    return (x > y) ? x : y;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T x, T y) {
    if (std::isnan(x)) {
      return x;
    }
    return (x > y) ? x : y;
  }
};

struct Minimum {
  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T x, T y) {
    return x < y ? x : y;
  }

  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T x, T y) {
    if (std::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }
};

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    constexpr float inf = std::numeric_limits<float>::infinity();
    auto maxval = Maximum()(x, y);
    auto minval = Minimum()(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : static_cast<decltype(x)>(
              maxval + std::log1p(fast_exp(minval - maxval)));
  }
};

struct Multiply {
  template <typename T>
  T operator()(T x, T y) {
    return x * y;
  }
};

struct NotEqual {
  template <typename T>
  bool operator()(T x, T y) {
    return x != y;
  }
};

struct Power {
  template <typename T>
  std::enable_if_t<!std::is_integral_v<T>, T> operator()(T base, T exp) {
    return std::pow(base, exp);
  }

  template <typename T>
  std::enable_if_t<std::is_integral_v<T>, T> operator()(T base, T exp) {
    T res = 1;
    while (exp) {
      if (exp & 1) {
        res *= base;
      }
      exp >>= 1;
      base *= base;
    }
    return res;
  }
};

struct Subtract {
  template <typename T>
  T operator()(T x, T y) {
    return x - y;
  }
};

struct LogicalAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x && y;
  }
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  }
};

struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};

struct BitwiseAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x & y;
  }
};

struct BitwiseOr {
  template <typename T>
  T operator()(T x, T y) {
    return x | y;
  }
};

struct BitwiseXor {
  template <typename T>
  T operator()(T x, T y) {
    return x ^ y;
  }
};

struct LeftShift {
  template <typename T>
  T operator()(T x, T y) {
    return x << y;
  }
};

struct RightShift {
  template <typename T>
  T operator()(T x, T y) {
    return x >> y;
  }
};

} // namespace mlx::core::detail
