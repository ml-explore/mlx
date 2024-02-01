// Copyright © 2023-24 Apple Inc.

namespace mlx::core::metal {

const char* get_kernel_preamble() {
  return R"preamble(
// Utils

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == metal::numeric_limits<float>::infinity()) {
    return metal::numeric_limits<float>::infinity();
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}


// Unary ops

float erf(float a) {
  float r, s, t, u;
  t = metal::abs(a);
  s = a * a;
  if (t > 0.927734375f) {
    // maximum error 0.99527 ulp
    r = metal::fma(
        -1.72853470e-5f, t, 3.83197126e-4f); // -0x1.220000p-16,0x1.91cfb2p-12
    u = metal::fma(
        -3.88396438e-3f, t, 2.42546219e-2f); // -0x1.fd1438p-9, 0x1.8d6342p-6
    r = metal::fma(r, s, u);
    r = metal::fma(r, t, -1.06777877e-1f); // -0x1.b55cb8p-4
    r = metal::fma(r, t, -6.34846687e-1f); // -0x1.450aa0p-1
    r = metal::fma(r, t, -1.28717512e-1f); // -0x1.079d0cp-3
    r = metal::fma(r, t, -t);
    // TODO, replace with expm1 when implemented
    r = 1.0f - metal::exp(r);
    r = metal::copysign(r, a);
  } else {
    // maximum error 0.98929 ulp
    r = -5.96761703e-4f; // -0x1.38e000p-11
    r = metal::fma(r, s, 4.99119423e-3f); //  0x1.471a58p-8
    r = metal::fma(r, s, -2.67681349e-2f); // -0x1.b691b2p-6
    r = metal::fma(r, s, 1.12819925e-1f); //  0x1.ce1c44p-4
    r = metal::fma(r, s, -3.76125336e-1f); // -0x1.812700p-2
    r = metal::fma(r, s, 1.28379166e-1f); //  0x1.06eba8p-3
    r = metal::fma(r, a, a);
  }
  return r;
}

float erfinv(float a) {
  auto t = metal::fma(a, 0.0f - a, 1.0f);
  t = metal::log(t);
  float p;
  if (metal::abs(t) > 6.125f) { // maximum ulp error = 2.35793
    p = 3.03697567e-10f; //  0x1.4deb44p-32
    p = metal::fma(p, t, 2.93243101e-8f); //  0x1.f7c9aep-26
    p = metal::fma(p, t, 1.22150334e-6f); //  0x1.47e512p-20
    p = metal::fma(p, t, 2.84108955e-5f); //  0x1.dca7dep-16
    p = metal::fma(p, t, 3.93552968e-4f); //  0x1.9cab92p-12
    p = metal::fma(p, t, 3.02698812e-3f); //  0x1.8cc0dep-9
    p = metal::fma(p, t, 4.83185798e-3f); //  0x1.3ca920p-8
    p = metal::fma(p, t, -2.64646143e-1f); // -0x1.0eff66p-2
    p = metal::fma(p, t, 8.40016484e-1f); //  0x1.ae16a4p-1
  } else { // maximum ulp error = 2.35002
    p = 5.43877832e-9f; //  0x1.75c000p-28
    p = metal::fma(p, t, 1.43285448e-7f); //  0x1.33b402p-23
    p = metal::fma(p, t, 1.22774793e-6f); //  0x1.499232p-20
    p = metal::fma(p, t, 1.12963626e-7f); //  0x1.e52cd2p-24
    p = metal::fma(p, t, -5.61530760e-5f); // -0x1.d70bd0p-15
    p = metal::fma(p, t, -1.47697632e-4f); // -0x1.35be90p-13
    p = metal::fma(p, t, 2.31468678e-3f); //  0x1.2f6400p-9
    p = metal::fma(p, t, 1.15392581e-2f); //  0x1.7a1e50p-7
    p = metal::fma(p, t, -2.32015476e-1f); // -0x1.db2aeep-3
    p = metal::fma(p, t, 8.86226892e-1f); //  0x1.c5bf88p-1
  }
  return a * p;
}

struct Abs {
  template <typename T> T operator()(T x) { return metal::abs(x); };
  template <> uint8_t operator()(uint8_t x) { return x; };
  template <> uint16_t operator()(uint16_t x) { return x; };
  template <> uint32_t operator()(uint32_t x) { return x; };
  template <> uint64_t operator()(uint64_t x) { return x; };
  template <> bool operator()(bool x) { return x; };
};

struct ArcCos {
  template <typename T> T operator()(T x) { return metal::precise::acos(x); };
};

struct ArcCosh {
  template <typename T> T operator()(T x) { return metal::precise::acosh(x); };
};

struct ArcSin {
  template <typename T> T operator()(T x) { return metal::precise::asin(x); };
};

struct ArcSinh {
  template <typename T> T operator()(T x) { return metal::precise::asinh(x); };
};

struct ArcTan {
  template <typename T> T operator()(T x) { return metal::precise::atan(x); };
};

struct ArcTanh {
  template <typename T> T operator()(T x) { return metal::precise::atanh(x); };
};

struct Ceil {
  template <typename T> T operator()(T x) { return metal::ceil(x); };
  template <> int8_t operator()(int8_t x) { return x; };
  template <> int16_t operator()(int16_t x) { return x; };
  template <> int32_t operator()(int32_t x) { return x; };
  template <> int64_t operator()(int64_t x) { return x; };
  template <> uint8_t operator()(uint8_t x) { return x; };
  template <> uint16_t operator()(uint16_t x) { return x; };
  template <> uint32_t operator()(uint32_t x) { return x; };
  template <> uint64_t operator()(uint64_t x) { return x; };
  template <> bool operator()(bool x) { return x; };
};

struct Cos {
  template <typename T> T operator()(T x) { return metal::precise::cos(x); };
};

struct Cosh {
  template <typename T> T operator()(T x) { return metal::precise::cosh(x); };
};

struct Erf {
  template <typename T> T operator()(T x) { return static_cast<T>(erf(static_cast<float>(x))); };
};

struct ErfInv {
  template <typename T> T operator()(T x) { return static_cast<T>(erfinv(static_cast<float>(x))); };
};

struct Exp {
  template <typename T> T operator()(T x) { return metal::precise::exp(x); };
};

struct Floor {
  template <typename T> T operator()(T x) { return metal::floor(x); };
  template <> int8_t operator()(int8_t x) { return x; };
  template <> int16_t operator()(int16_t x) { return x; };
  template <> int32_t operator()(int32_t x) { return x; };
  template <> int64_t operator()(int64_t x) { return x; };
  template <> uint8_t operator()(uint8_t x) { return x; };
  template <> uint16_t operator()(uint16_t x) { return x; };
  template <> uint32_t operator()(uint32_t x) { return x; };
  template <> uint64_t operator()(uint64_t x) { return x; };
  template <> bool operator()(bool x) { return x; };
};

struct Log {
  template <typename T> T operator()(T x) { return metal::precise::log(x); };
};

struct Log2 {
  template <typename T> T operator()(T x) { return metal::precise::log2(x); };
};

struct Log10 {
  template <typename T> T operator()(T x) { return metal::precise::log10(x); };
};

struct Log1p {
  template <typename T> T operator()(T x) { return log1p(x); };
};

struct LogicalNot {
  template <typename T> T operator()(T x) { return !x; };
};

struct Negative {
  template <typename T> T operator()(T x) { return -x; };
};

struct Round {
  template <typename T> T operator()(T x) { return metal::rint(x); };
};

struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T> T operator()(T x) { return (x > T(0)) - (x < T(0)); };
  template <> uint32_t operator()(uint32_t x) { return x != 0; };
};

struct Sin {
  template <typename T> T operator()(T x) { return metal::precise::sin(x); };
};

struct Sinh {
  template <typename T> T operator()(T x) { return metal::precise::sinh(x); };
};

struct Square {
  template <typename T> T operator()(T x) { return x * x; };
};

struct Sqrt {
  template <typename T> T operator()(T x) { return metal::precise::sqrt(x); };
};

struct Rsqrt {
  template <typename T> T operator()(T x) { return metal::precise::rsqrt(x); };
};

struct Tan {
  template <typename T> T operator()(T x) { return metal::precise::tan(x); };
};

struct Tanh {
  template <typename T> T operator()(T x) { return metal::precise::tanh(x); };
};

// Binary ops
struct Add {
  template <typename T> T operator()(T x, T y) { return x + y; }
};

struct Divide {
  template <typename T> T operator()(T x, T y) { return x / y; }
};

struct Remainder {
  template <typename T> T operator()(T x, T y) { return x % y; }
  template <> float operator()(float x, float y) { return metal::fmod(x, y); }
  template <> half operator()(half x, half y) { return metal::fmod(x, y); }
};

struct Equal {
  template <typename T> bool operator()(T x, T y) { return x == y; }
};

struct NaNEqual {
  template <typename T> bool operator()(T x, T y) {
    return x == y || (metal::isnan(x) && metal::isnan(y));
  }
};

struct Greater {
  template <typename T> bool operator()(T x, T y) { return x > y; }
};

struct GreaterEqual {
  template <typename T> bool operator()(T x, T y) { return x >= y; }
};

struct Less {
  template <typename T> bool operator()(T x, T y) { return x < y; }
};

struct LessEqual {
  template <typename T> bool operator()(T x, T y) { return x <= y; }
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
    return (minval == -inf || maxval == inf) ? maxval :
      (maxval + log1p(metal::exp(minval - maxval)));
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
};

struct Multiply {
  template <typename T> T operator()(T x, T y) { return x * y; }
};

struct NotEqual {
  template <typename T> bool operator()(T x, T y) { return x != y; }
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
};


struct Subtract {
  template <typename T> T operator()(T x, T y) { return x - y; }
};

struct LogicalAnd {
    template <typename T>
    T operator()(T x, T y) { return x && y; };
};

struct LogicalOr {
    template <typename T>
    T operator()(T x, T y) { return x || y; };
};
  )preamble";
}

} // namespace mlx::core::metal
