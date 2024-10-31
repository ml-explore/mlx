// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_integer>
#include <metal_math>

struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
  }
};

struct FloorDivide {
  template <typename T>
  T operator()(T x, T y) {
    return x / y;
  }
  template <>
  float operator()(float x, float y) {
    return trunc(x / y);
  }
  template <>
  half operator()(half x, half y) {
    return trunc(x / y);
  }
  template <>
  bfloat16_t operator()(bfloat16_t x, bfloat16_t y) {
    return trunc(x / y);
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
  metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    return x % y;
  }
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T> & metal::is_signed_v<T>, T>
  operator()(T x, T y) {
    auto r = x % y;
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    T r = fmod(x, y);
    if (r != 0 && (r < 0 != y < 0)) {
      r += y;
    }
    return r;
  }
  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    return x % y;
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
    return x == y || (metal::isnan(x) && metal::isnan(y));
  }
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x == y ||
        (metal::isnan(x.real) && metal::isnan(y.real) && metal::isnan(x.imag) &&
         metal::isnan(y.imag)) ||
        (x.real == y.real && metal::isnan(x.imag) && metal::isnan(y.imag)) ||
        (metal::isnan(x.real) && metal::isnan(y.real) && x.imag == y.imag);
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

struct LogAddExp {
  template <typename T>
  T operator()(T x, T y) {
    if (metal::isnan(x) || metal::isnan(y)) {
      return metal::numeric_limits<T>::quiet_NaN();
    }
    constexpr T inf = metal::numeric_limits<T>::infinity();
    T maxval = metal::max(x, y);
    T minval = metal::min(x, y);
    return (minval == -inf || maxval == inf)
        ? maxval
        : (maxval + log1p(metal::exp(minval - maxval)));
  };
};

struct Maximum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::max(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x > y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x > y ? x : y;
  }
};

struct Minimum {
  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T x, T y) {
    return metal::min(x, y);
  }

  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T x, T y) {
    if (metal::isnan(x)) {
      return x;
    }
    return x < y ? x : y;
  }

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    if (metal::isnan(x.real) || metal::isnan(x.imag)) {
      return x;
    }
    return x < y ? x : y;
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
  template <>
  bool operator()(complex64_t x, complex64_t y) {
    return x.real != y.real || x.imag != y.imag;
  }
};

struct Power {
  template <typename T>
  metal::enable_if_t<!metal::is_integral_v<T>, T> operator()(T base, T exp) {
    return metal::pow(base, exp);
  }

  template <typename T>
  metal::enable_if_t<metal::is_integral_v<T>, T> operator()(T base, T exp) {
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

  template <>
  complex64_t operator()(complex64_t x, complex64_t y) {
    auto x_theta = metal::atan2(x.imag, x.real);
    auto x_ln_r = 0.5 * metal::log(x.real * x.real + x.imag * x.imag);
    auto mag = metal::exp(y.real * x_ln_r - y.imag * x_theta);
    auto phase = y.imag * x_ln_r + y.real * x_theta;
    return {mag * metal::cos(phase), mag * metal::sin(phase)};
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
  };
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  };
};

struct BitwiseAnd {
  template <typename T>
  T operator()(T x, T y) {
    return x & y;
  };
};

struct BitwiseOr {
  template <typename T>
  T operator()(T x, T y) {
    return x | y;
  };
};

struct BitwiseXor {
  template <typename T>
  T operator()(T x, T y) {
    return x ^ y;
  };
};

struct LeftShift {
  template <typename T>
  T operator()(T x, T y) {
    return x << y;
  };
};

struct RightShift {
  template <typename T>
  T operator()(T x, T y) {
    return x >> y;
  };
};

struct ArcTan2 {
  template <typename T>
  T operator()(T y, T x) {
    return metal::precise::atan2(y, x);
  }
};

struct DivMod {
  template <typename T>
  metal::array<T, 2> operator()(T x, T y) {
    return {FloorDivide{}(x, y), Remainder{}(x, y)};
  };
};
