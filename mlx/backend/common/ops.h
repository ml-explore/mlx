// Copyright © 2023-2024 Apple Inc.

#pragma once
#include <stdint.h>
#include <cmath>
#include <complex>

namespace mlx::core::detail {

namespace {
constexpr float inf = std::numeric_limits<float>::infinity();
} // namespace

typedef union {
  int i;
  float f;
} IntOrFloat;

inline float fast_exp(float x) {
  if (x == -std::numeric_limits<float>::infinity()) {
    return 0.0f;
  } else if (x == std::numeric_limits<float>::infinity() || std::isnan(x)) {
    return x;
  }
  x *= 1.442695; // multiply with log_2(e)
  float ipart, fpart;
  IntOrFloat epart;
  x = std::max(-80.f, std::min(x, 80.f));
  ipart = std::floor(x + 0.5);
  fpart = x - ipart;

  x = 1.535336188319500e-4f;
  x = x * fpart + 1.339887440266574e-3f;
  x = x * fpart + 9.618437357674640e-3f;
  x = x * fpart + 5.550332471162809e-2f;
  x = x * fpart + 2.402264791363012e-1f;
  x = x * fpart + 6.931472028550421e-1f;
  x = x * fpart + 1.000000000000000f;

  // generate 2**ipart in the floating point representation using integer
  // bitshifting
  epart.i = (int(ipart) + 127) << 23;

  return epart.f * x;
}

inline float fast_erf(float a) {
  float r, s, t, u;
  t = std::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = std::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = std::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = std::fma(r, s, u);
    r = std::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = std::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = std::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = std::fma(r, t, -t);
    // TODO, replace with expm1 when implemented
    r = 1.0f - std::exp(r);
    r = std::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = std::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = std::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = std::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = std::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = std::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = std::fma(r, a, a);
  }
  return r;
}

inline float fast_erfinv(float a) {
  auto t = std::fma(a, 0.0f - a, 1.0f);
  t = std::log(t);
  float p;
  if (std::abs(t) > 6.125f) { // maximum ulp error = 2.35793
    p = 3.03697567e-10f; //  0x1.4deb44p-32
    p = std::fma(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26
    p = std::fma(p, t, 1.22150334e-6f); //  0x1.47e512p-20
    p = std::fma(p, t, 2.84108955e-5f); //  0x1.dca7dep-16
    p = std::fma(p, t, 3.93552968e-4f); //  0x1.9cab92p-12
    p = std::fma(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9
    p = std::fma(p, t, 4.83185798e-3f); //  0x1.3ca920p-8
    p = std::fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = std::fma(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1
  } else { // maximum ulp error = 2.35002
    p = 5.43877832e-9f; //  0x1.75c000p-28
    p = std::fma(p, t, 1.43285448e-7f); //  0x1.33b402p-23
    p = std::fma(p, t, 1.22774793e-6f); //  0x1.499232p-20
    p = std::fma(p, t, 1.12963626e-7f); //  0x1.e52cd2p-24
    p = std::fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = std::fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = std::fma(p, t, 2.31468678e-3f); //  0x1.2f6400p-9
    p = std::fma(p, t, 1.15392581e-2f); //  0x1.7a1e50p-7
    p = std::fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = std::fma(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1
  }
  return a * p;
}

struct Abs {
  template <typename T>
  T operator()(T x) {
    return std::abs(x);
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct ArcCos {
  template <typename T>
  T operator()(T x) {
    return std::acos(x);
  };
};

struct ArcCosh {
  template <typename T>
  T operator()(T x) {
    return std::acosh(x);
  };
};

struct ArcSin {
  template <typename T>
  T operator()(T x) {
    return std::asin(x);
  };
};

struct ArcSinh {
  template <typename T>
  T operator()(T x) {
    return std::asinh(x);
  };
};

struct ArcTan {
  template <typename T>
  T operator()(T x) {
    return std::atan(x);
  };
};

struct ArcTanh {
  template <typename T>
  T operator()(T x) {
    return std::atanh(x);
  };
};

struct Ceil {
  template <typename T>
  T operator()(T x) {
    return std::ceil(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct Cos {
  template <typename T>
  T operator()(T x) {
    return std::cos(x);
  };
};

struct Cosh {
  template <typename T>
  T operator()(T x) {
    return std::cosh(x);
  };
};

struct Erf {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erf(static_cast<float>(x)));
  };
};

struct ErfInv {
  template <typename T>
  T operator()(T x) {
    return static_cast<T>(fast_erfinv(static_cast<float>(x)));
  };
};

struct Exp {
  template <typename T>
  T operator()(T x) {
    return fast_exp(x);
  };

  complex64_t operator()(complex64_t x) {
    return std::exp(x);
  }
};

struct Floor {
  template <typename T>
  T operator()(T x) {
    return std::floor(x);
  };
  int8_t operator()(int8_t x) {
    return x;
  };
  int16_t operator()(int16_t x) {
    return x;
  };
  int32_t operator()(int32_t x) {
    return x;
  };
  int64_t operator()(int64_t x) {
    return x;
  };
  uint8_t operator()(uint8_t x) {
    return x;
  };
  uint16_t operator()(uint16_t x) {
    return x;
  };
  uint32_t operator()(uint32_t x) {
    return x;
  };
  uint64_t operator()(uint64_t x) {
    return x;
  };
  bool operator()(bool x) {
    return x;
  };
};

struct Log {
  template <typename T>
  T operator()(T x) {
    return std::log(x);
  };
};

struct Log2 {
  template <typename T>
  T operator()(T x) {
    return std::log2(x);
  };
};

struct Log10 {
  template <typename T>
  T operator()(T x) {
    return std::log10(x);
  };
};

struct Log1p {
  template <typename T>
  T operator()(T x) {
    return log1p(x);
  };
};

struct LogicalNot {
  template <typename T>
  T operator()(T x) {
    return !x;
  };
};

struct Negative {
  template <typename T>
  T operator()(T x) {
    return -x;
  };
};

struct Round {
  template <typename T>
  T operator()(T x) {
    return std::rint(x);
  }

  complex64_t operator()(complex64_t x) {
    return {std::rint(x.real()), std::rint(x.imag())};
  }
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto one = static_cast<decltype(x)>(1.0);
    return one / (one + fast_exp(-x));
  }
};

struct Sign {
  template <typename T>
  T operator()(T x) {
    return (x > T(0)) - (x < T(0));
  }
  uint8_t operator()(uint8_t x) {
    return x != 0;
  }
  uint16_t operator()(uint16_t x) {
    return x != 0;
  }
  uint32_t operator()(uint32_t x) {
    return x != 0;
  }
  uint64_t operator()(uint64_t x) {
    return x != 0;
  }
};

struct Sin {
  template <typename T>
  T operator()(T x) {
    return std::sin(x);
  };
};

struct Sinh {
  template <typename T>
  T operator()(T x) {
    return std::sinh(x);
  };
};

struct Square {
  template <typename T>
  T operator()(T x) {
    return x * x;
  };
};

struct Sqrt {
  template <typename T>
  T operator()(T x) {
    return std::sqrt(x);
  };
};

struct Rsqrt {
  template <typename T>
  T operator()(T x) {
    return static_cast<decltype(x)>(1.0) / std::sqrt(x);
  };
};

struct Tan {
  template <typename T>
  T operator()(T x) {
    return std::tan(x);
  };
};

struct Tanh {
  template <typename T>
  T operator()(T x) {
    return std::tanh(x);
  };
};

struct Add {
  template <typename T>
  T operator()(T x, T y) {
    return x + y;
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
    return x == y || (std::isnan(x) && std::isnan(y));
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
  };
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
  };
};

struct LogicalOr {
  template <typename T>
  T operator()(T x, T y) {
    return x || y;
  };
};

struct Select {
  template <typename T>
  T operator()(bool condition, T x, T y) {
    return condition ? x : y;
  }
};

} // namespace mlx::core::detail
