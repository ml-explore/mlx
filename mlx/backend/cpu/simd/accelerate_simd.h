#pragma once

#include <simd/math.h>
#include <simd/vector.h>

#include <stdint.h>
#include <cmath>
#include <complex>

#include "mlx/backend/cpu/simd/base_simd.h"

// There seems to be a bug in sims/base.h
// __XROS_2_0 is not defined, the expression evaluates
// to true instead of false setting the SIMD library
// higher than it should be even on macOS < 15
#if __MAC_OS_X_VERSION_MIN_REQUIRED >= 150000 ||  \
    __IPHONE_OS_VERSION_MIN_REQUIRED >= 180000 || \
    __WATCH_OS_VERSION_MIN_REQUIRED >= 110000 ||  \
    __WATCH_OS_VERSION_MIN_REQUIRED >= 110000 ||  \
    __TV_OS_VERSION_MIN_REQUIRED >= 180000
#define MLX_SIMD_LIBRARY_VERSION 6
#else
#define MLX_SIMD_LIBRARY_VERSION 5
#endif

namespace mlx::core::simd {

// Apple simd namespace
namespace asd = ::simd;

// This indirection is needed to remap certain types to ones that accelerate
// SIMD can handle
template <typename T, int N>
struct ScalarT {
  using v = T;
};
template <int N>
struct ScalarT<bool, N> {
  using v = char;
};
template <int N>
struct ScalarT<int8_t, N> {
  using v = char;
};
template <int N>
struct ScalarT<uint64_t, N> {
  using v = unsigned long;
};
template <int N>
struct ScalarT<int64_t, N> {
  using v = long;
};

template <typename T, int N>
struct Simd {
  static constexpr int size = N;
  using scalar_t = typename ScalarT<T, N>::v;

  Simd<T, N>() {}

  template <typename U>
  Simd<T, N>(Simd<U, N> other) : value(asd::convert<scalar_t>(other.value)) {}

  template <typename U>
  Simd<T, N>(U v) : value(v){};

  Simd<T, N>(Simd<T, N / 2> x, Simd<T, N / 2> y) {
    value = asd::make<typename asd::Vector<scalar_t, N>::packed_t>(
        x.value, y.value);
  };

  T operator[](int idx) const {
    return reinterpret_cast<const T*>(&value)[idx];
  }

  T& operator[](int idx) {
    return reinterpret_cast<T*>(&value)[idx];
  }

  typename asd::Vector<scalar_t, N>::packed_t value;
};

// Values chosen based on benchmarks on M3 Max
// TODO: consider choosing these more optimally
template <>
inline constexpr int max_size<int8_t> = 16;
template <>
inline constexpr int max_size<int16_t> = 16;
template <>
inline constexpr int max_size<int> = 8;
template <>
inline constexpr int max_size<int64_t> = 4;
template <>
inline constexpr int max_size<uint8_t> = 16;
template <>
inline constexpr int max_size<uint16_t> = 16;
template <>
inline constexpr int max_size<uint32_t> = 8;
template <>
inline constexpr int max_size<uint64_t> = 4;
template <>
inline constexpr int max_size<float> = 8;
template <>
inline constexpr int max_size<double> = 4;

#define SIMD_DEFAULT_UNARY(name, op) \
  template <typename T, int N>       \
  Simd<T, N> name(Simd<T, N> v) {    \
    return op(v.value);              \
  }

SIMD_DEFAULT_UNARY(abs, asd::abs)
SIMD_DEFAULT_UNARY(floor, asd::floor)
SIMD_DEFAULT_UNARY(acos, asd::acos)
SIMD_DEFAULT_UNARY(acosh, asd::acosh)
SIMD_DEFAULT_UNARY(asin, asd::asin)
SIMD_DEFAULT_UNARY(asinh, asd::asinh)
SIMD_DEFAULT_UNARY(atan, asd::atan)
SIMD_DEFAULT_UNARY(atanh, asd::atanh)
SIMD_DEFAULT_UNARY(ceil, asd::ceil)
SIMD_DEFAULT_UNARY(cosh, asd::cosh)
SIMD_DEFAULT_UNARY(expm1, asd::expm1)
SIMD_DEFAULT_UNARY(log, asd::log)
SIMD_DEFAULT_UNARY(log2, asd::log2)
SIMD_DEFAULT_UNARY(log10, asd::log10)
SIMD_DEFAULT_UNARY(log1p, asd::log1p)
SIMD_DEFAULT_UNARY(rint, asd::rint)
SIMD_DEFAULT_UNARY(sinh, asd::sinh)
SIMD_DEFAULT_UNARY(sqrt, asd::sqrt)
SIMD_DEFAULT_UNARY(rsqrt, asd::rsqrt)
SIMD_DEFAULT_UNARY(recip, asd::recip)
SIMD_DEFAULT_UNARY(tan, asd::tan)
SIMD_DEFAULT_UNARY(tanh, asd::tanh)

template <typename T, int N>
Simd<T, N> operator-(Simd<T, N> v) {
  return -v.value;
}

template <typename T, int N>
Simd<T, N> operator~(Simd<T, N> v) {
  return ~v.value;
}

template <typename T, int N>
Simd<bool, N> isnan(Simd<T, N> v) {
  return asd::convert<char>(v.value != v.value);
}

// No simd_boolN in accelerate, use int8_t instead
template <typename T, int N>
Simd<bool, N> operator!(Simd<T, N> v) {
  return asd::convert<char>(!v.value);
}

#define SIMD_DEFAULT_BINARY(OP)                                              \
  template <typename T, typename U, int N>                                   \
  Simd<T, N> operator OP(Simd<T, N> x, U y) {                                \
    return asd::convert<typename Simd<T, N>::scalar_t>(x.value OP y);        \
  }                                                                          \
  template <typename T1, typename T2, int N>                                 \
  Simd<T2, N> operator OP(T1 x, Simd<T2, N> y) {                             \
    return asd::convert<typename Simd<T2, N>::scalar_t>(x OP y.value);       \
  }                                                                          \
  template <typename T1, typename T2, int N>                                 \
  Simd<T1, N> operator OP(Simd<T1, N> x, Simd<T2, N> y) {                    \
    return asd::convert<typename Simd<T1, N>::scalar_t>(x.value OP y.value); \
  }

SIMD_DEFAULT_BINARY(+)
SIMD_DEFAULT_BINARY(-)
SIMD_DEFAULT_BINARY(/)
SIMD_DEFAULT_BINARY(*)
SIMD_DEFAULT_BINARY(<<)
SIMD_DEFAULT_BINARY(>>)
SIMD_DEFAULT_BINARY(|)
SIMD_DEFAULT_BINARY(^)
SIMD_DEFAULT_BINARY(&)
SIMD_DEFAULT_BINARY(&&)
SIMD_DEFAULT_BINARY(||)

#define SIMD_DEFAULT_COMPARISONS(OP)                        \
  template <int N, typename T, typename U>                  \
  Simd<bool, N> operator OP(Simd<T, N> a, U b) {            \
    return asd::convert<char>(a.value OP b);                \
  }                                                         \
  template <int N, typename T, typename U>                  \
  Simd<bool, N> operator OP(T a, Simd<U, N> b) {            \
    return asd::convert<char>(a OP b.value);                \
  }                                                         \
  template <int N, typename T1, typename T2>                \
  Simd<bool, N> operator OP(Simd<T1, N> a, Simd<T2, N> b) { \
    return asd::convert<char>(a.value OP b.value);          \
  }

SIMD_DEFAULT_COMPARISONS(>)
SIMD_DEFAULT_COMPARISONS(<)
SIMD_DEFAULT_COMPARISONS(>=)
SIMD_DEFAULT_COMPARISONS(<=)
SIMD_DEFAULT_COMPARISONS(==)
SIMD_DEFAULT_COMPARISONS(!=)

template <typename T, int N>
Simd<T, N> atan2(Simd<T, N> a, Simd<T, N> b) {
  return asd::atan2(a.value, b.value);
}

template <typename T, int N>
Simd<T, N> maximum(Simd<T, N> a, Simd<T, N> b) {
  // TODO add isnan
  return asd::max(a.value, b.value);
}

template <typename T, int N>
Simd<T, N> minimum(Simd<T, N> a, Simd<T, N> b) {
  // TODO add isnan
  return asd::min(a.value, b.value);
}

template <typename T, int N>
Simd<T, N> remainder(Simd<T, N> a, Simd<T, N> b) {
  Simd<T, N> r;
  if constexpr (!std::is_integral_v<T>) {
    r = asd::remainder(a.value, b.value);
  } else {
    r = a - b * (a / b);
  }
  if constexpr (std::is_signed_v<T>) {
    auto mask = r != 0 && (r < 0 != b < 0);
    r = select(mask, r + b, r);
  }
  return r;
}

template <typename MaskT, typename T1, typename T2, int N>
Simd<T1, N> select(Simd<MaskT, N> mask, Simd<T1, N> x, Simd<T2, N> y) {
  if constexpr (sizeof(T1) == 1) {
    return asd::bitselect(y.value, x.value, asd::convert<char>(mask.value));
  } else if constexpr (sizeof(T1) == 2) {
    return asd::bitselect(y.value, x.value, asd::convert<short>(mask.value));
  } else if constexpr (sizeof(T1) == 4) {
    return asd::bitselect(y.value, x.value, asd::convert<int>(mask.value));
  } else {
    return asd::bitselect(y.value, x.value, asd::convert<long>(mask.value));
  }
}

template <typename T, int N>
Simd<T, N> pow(Simd<T, N> base, Simd<T, N> exp) {
  if constexpr (!std::is_integral_v<T>) {
    return asd::pow(base.value, exp.value);
  } else {
    Simd<T, N> res = 1;
    while (any(exp)) {
      res = select(exp & 1, res * base, res);
      base = select(exp, base * base, base);
      exp = exp >> 1;
    }
    return res;
  }
}

template <typename T, int N>
Simd<T, N> clamp(Simd<T, N> v, Simd<T, N> min, Simd<T, N> max) {
  return asd::clamp(v.value, min.value, max.value);
}

template <typename T, typename U, int N>
Simd<T, N> fma(Simd<T, N> x, Simd<T, N> y, U z) {
  return asd::muladd(x.value, y.value, Simd<T, N>(z).value);
}

// Reductions

template <typename T, int N>
bool all(Simd<T, N> x) {
  return asd::all(x.value);
}
template <typename T, int N>
bool any(Simd<T, N> x) {
  return asd::any(x.value);
}
template <typename T, int N>
T sum(Simd<T, N> x) {
  return asd::reduce_add(x.value);
}
template <typename T, int N>
T max(Simd<T, N> x) {
  return asd::reduce_max(x.value);
}
template <typename T, int N>
T min(Simd<T, N> x) {
  return asd::reduce_min(x.value);
}

template <typename T, int N>
T prod(Simd<T, N> x) {
  auto ptr = (T*)&x;
  auto lhs = load<T, N / 2>(ptr);
  auto rhs = load<T, N / 2>(ptr + N / 2);
  return prod(lhs * rhs);
}

} // namespace mlx::core::simd

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "mlx/backend/cpu/simd/accelerate_fp16_simd.h"
#endif
