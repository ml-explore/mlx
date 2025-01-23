#pragma once

#include <simd/math.h>
#include <simd/vector.h>

#include <stdint.h>
#include <cmath>
#include <complex>

#include "mlx/backend/common/simd/default_simd.h"

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

// TODO macOS 15+
template <int N>
struct ScalarT<__fp16, N> {
  using v = _Float16;
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

  T operator[](int idx) const {
    return reinterpret_cast<const float*>(&value)[idx];
  }

  T& operator[](int idx) {
    return reinterpret_cast<T*>(&value)[idx];
  }

  typename asd::Vector<scalar_t, N>::packed_t value;
};

// Values chosen based on benchmarks on M3 Max
// TODO: consider choosing these more optimally
template <>
static constexpr int max_size<int8_t> = 16;
template <>
static constexpr int max_size<int16_t> = 16;
template <>
static constexpr int max_size<int> = 8;
template <>
static constexpr int max_size<int64_t> = 4;
template <>
static constexpr int max_size<uint8_t> = 16;
template <>
static constexpr int max_size<uint16_t> = 16;
template <>
static constexpr int max_size<uint32_t> = 8;
template <>
static constexpr int max_size<uint64_t> = 4;
template <>
static constexpr int max_size<float> = 8;
template <>
static constexpr int max_size<double> = 4;

#define SIMD_DEFAULT_UNARY(name, op) \
  template <typename T, int N>       \
  Simd<T, N> name(Simd<T, N> v) {    \
    return op(v.value);              \
  }

SIMD_DEFAULT_UNARY(abs, asd::abs);
SIMD_DEFAULT_UNARY(floor, asd::floor);
SIMD_DEFAULT_UNARY(acos, asd::acos)
SIMD_DEFAULT_UNARY(acosh, asd::acosh)
SIMD_DEFAULT_UNARY(asin, asd::asin)
SIMD_DEFAULT_UNARY(asinh, asd::asinh)
SIMD_DEFAULT_UNARY(atan, asd::atan)
SIMD_DEFAULT_UNARY(atanh, asd::atanh)
SIMD_DEFAULT_UNARY(ceil, asd::ceil)
SIMD_DEFAULT_UNARY(cos, asd::cos);
SIMD_DEFAULT_UNARY(cosh, asd::cosh)
SIMD_DEFAULT_UNARY(erf, asd::erf)
SIMD_DEFAULT_UNARY(expm1, asd::expm1)
SIMD_DEFAULT_UNARY(log, asd::log)
SIMD_DEFAULT_UNARY(log2, asd::log2)
SIMD_DEFAULT_UNARY(log10, asd::log10)
SIMD_DEFAULT_UNARY(log1p, asd::log1p)
SIMD_DEFAULT_UNARY(rint, asd::rint)
SIMD_DEFAULT_UNARY(sin, asd::sin)
SIMD_DEFAULT_UNARY(sinh, asd::sinh)
SIMD_DEFAULT_UNARY(sqrt, asd::sqrt)
SIMD_DEFAULT_UNARY(tan, asd::tan)
SIMD_DEFAULT_UNARY(tanh, asd::tanh)

template <typename T, int N>
Simd<T, N> operator-(Simd<T, N> v) {
  return -v.value;
}

// No simd_boolN in accelerate, use int8_t instead
template <typename T, int N>
Simd<bool, N> operator!(Simd<T, N> v) {
  return asd::convert<char>(!v.value);
}

#define SIMD_DEFAULT_BINARY(OP)                           \
  template <typename T, typename U, int N>                \
  Simd<T, N> operator OP(Simd<T, N> x, U y) {             \
    return x.value OP y;                                  \
  }                                                       \
  template <typename T, typename U, int N>                \
  Simd<U, N> operator OP(T x, Simd<U, N> y) {             \
    return x OP y.value;                                  \
  }                                                       \
  template <typename T1, typename T2, int N>              \
  Simd<T1, N> operator OP(Simd<T1, N> x, Simd<T2, N> y) { \
    return x.value OP y.value;                            \
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
Simd<T, N> clamp(Simd<T, N> v, Simd<T, N> min, Simd<T, N> max) {
  return asd::clamp(v.value, min.value, max.value);
}

template <typename MaskT, typename T1, typename T2, int N>
Simd<T1, N> select(Simd<MaskT, N> mask, Simd<T1, N> x, Simd<T2, N> y) {
  return asd::bitselect(x.value, y.value, mask.value);
}

template <typename T, int N>
Simd<T, N> fma(Simd<T, N> x, Simd<T, N> y, T z) {
  return asd::fma(x.value, y.value, Simd<T, N>(z).value);
}

} // namespace mlx::core::simd
