#pragma once

#include <stdint.h>
#include <algorithm>
#include <cmath>
#include <complex>

#include "mlx/backend/common/simd/scalar_math.h"

namespace mlx::core::simd {
template <typename T, int N>
struct Simd;

template <typename T>
static constexpr int max_size = 1;

template <typename T>
struct Simd<T, 1> {
  static constexpr int size = 1;
  T value;
  Simd() {}
  template <typename U>
  Simd(Simd<U, 1> v) : value(v.value) {}
  template <typename U>
  Simd(U v) : value(v) {}
};

template <typename T, int N>
Simd<T, N> load(const T* x) {
  return *(Simd<T, N>*)x;
}

template <typename T, int N>
void store(T* dst, const Simd<T, N>& x) {
  *(Simd<T, N>*)dst = x;
}

template <typename, typename = void>
constexpr bool is_complex = false;

template <typename T>
constexpr bool is_complex<T, std::void_t<decltype(std::declval<T>().real())>> =
    true;

template <typename T>
Simd<T, 1> rint(Simd<T, 1> in) {
  if constexpr (is_complex<T>) {
    return Simd<T, 1>{
        T{std::rint(in.value.real()), std::rint(in.value.imag())}};
  } else {
    return Simd<T, 1>{std::rint(in.value)};
  }
}

#define DEFAULT_UNARY(name, op)                                 \
  template <typename T>                                         \
  auto name(Simd<T, 1> in) -> Simd<decltype(op(in.value)), 1> { \
    return op(in.value);                                        \
  }

DEFAULT_UNARY(operator-, std::negate{})
DEFAULT_UNARY(operator!, std::logical_not{})
DEFAULT_UNARY(abs, std::abs)
DEFAULT_UNARY(acos, std::acos)
DEFAULT_UNARY(acosh, std::acosh)
DEFAULT_UNARY(asin, std::asin)
DEFAULT_UNARY(asinh, std::asinh)
DEFAULT_UNARY(atan, std::atan)
DEFAULT_UNARY(atanh, std::atanh)
DEFAULT_UNARY(ceil, std::ceil)
DEFAULT_UNARY(conj, std::conj)
DEFAULT_UNARY(cos, std::cos)
DEFAULT_UNARY(cosh, std::cosh)
DEFAULT_UNARY(erf, erf);
DEFAULT_UNARY(erfinv, erfinv);
DEFAULT_UNARY(expm1, std::expm1)
DEFAULT_UNARY(floor, std::floor)
DEFAULT_UNARY(imag, std::imag)
DEFAULT_UNARY(log, std::log)
DEFAULT_UNARY(log2, std::log2)
DEFAULT_UNARY(log10, std::log10)
DEFAULT_UNARY(log1p, std::log1p)
DEFAULT_UNARY(real, std::real)
DEFAULT_UNARY(sin, std::sin)
DEFAULT_UNARY(sinh, std::sinh)
DEFAULT_UNARY(sqrt, std::sqrt)
DEFAULT_UNARY(tan, std::tan)
DEFAULT_UNARY(tanh, std::tanh)
DEFAULT_UNARY(isnan, std::isnan)

#define DEFAULT_BINARY(OP)                                                 \
  template <typename T1, typename T2>                                      \
  auto operator OP(Simd<T1, 1> a, Simd<T2, 1> b)                           \
      ->Simd<decltype(a.value OP b.value), 1> {                            \
    return a.value OP b.value;                                             \
  }                                                                        \
  template <typename T1, typename T2>                                      \
  auto operator OP(T1 a, Simd<T2, 1> b)->Simd<decltype(a OP b.value), 1> { \
    return a OP b.value;                                                   \
  }                                                                        \
  template <typename T1, typename T2>                                      \
  auto operator OP(Simd<T1, 1> a, T2 b)->Simd<decltype(a.value OP b), 1> { \
    return a.value OP b;                                                   \
  }

DEFAULT_BINARY(+)
DEFAULT_BINARY(-)
DEFAULT_BINARY(*)
DEFAULT_BINARY(/)
DEFAULT_BINARY(<<)
DEFAULT_BINARY(>>)
DEFAULT_BINARY(|)
DEFAULT_BINARY(^)
DEFAULT_BINARY(&)
DEFAULT_BINARY(&&)
DEFAULT_BINARY(||)

template <typename T>
Simd<T, 1> remainder(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  T r;
  if constexpr (std::is_integral_v<T>) {
    r = a % b;
  } else {
    r = std::remainder(a, b);
  }
  if constexpr (std::is_signed_v<T>) {
    if (r != 0 && (r < 0 != b < 0)) {
      r += b;
    }
  }
  return r;
}

template <typename T>
Simd<T, 1> maximum(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  if constexpr (!std::is_integral_v<T>) {
    if (std::isnan(a)) {
      return a;
    }
  }
  return (a > b) ? a : b;
}

template <typename T>
Simd<T, 1> minimum(Simd<T, 1> a_, Simd<T, 1> b_) {
  T a = a_.value;
  T b = b_.value;
  if constexpr (!std::is_integral_v<T>) {
    if (std::isnan(a)) {
      return a;
    }
  }
  return (a < b) ? a : b;
}

template <typename T>
Simd<T, 1> pow(Simd<T, 1> a, Simd<T, 1> b) {
  T base = a.value;
  T exp = b.value;
  if constexpr (!std::is_integral_v<T>) {
    return std::pow(base, exp);
  } else {
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
}

template <typename T>
Simd<T, 1> atan2(Simd<T, 1> a, Simd<T, 1> b) {
  return std::atan2(a.value, b.value);
}

#define DEFAULT_COMPARISONS(OP)                             \
  template <typename T1, typename T2>                       \
  Simd<bool, 1> operator OP(Simd<T1, 1> a, Simd<T2, 1> b) { \
    return a.value OP b.value;                              \
  }                                                         \
  template <typename T1, typename T2>                       \
  Simd<bool, 1> operator OP(T1 a, Simd<T2, 1> b) {          \
    return a OP b.value;                                    \
  }                                                         \
  template <typename T1, typename T2>                       \
  Simd<bool, 1> operator OP(Simd<T1, 1> a, T2 b) {          \
    return a.value OP b;                                    \
  }

DEFAULT_COMPARISONS(>)
DEFAULT_COMPARISONS(<)
DEFAULT_COMPARISONS(>=)
DEFAULT_COMPARISONS(<=)
DEFAULT_COMPARISONS(==)
DEFAULT_COMPARISONS(!=)

template <typename MaskT, typename T>
Simd<T, 1> select(Simd<MaskT, 1> mask, Simd<T, 1> x, Simd<T, 1> y) {
  return mask.value ? x.value : y.value;
}

template <typename T>
Simd<T, 1> clamp(Simd<T, 1> v, Simd<T, 1> min, Simd<T, 1> max) {
  return std::clamp(v.value, min.value, max.value);
}

template <typename T>
Simd<T, 1> fma(Simd<T, 1> x, Simd<T, 1> y, T z) {
  return std::fma(x.value, y.value, z);
}
} // namespace mlx::core::simd
