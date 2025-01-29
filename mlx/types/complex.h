// Copyright © 2023 Apple Inc.

#pragma once
#include <complex>
#include "mlx/types/half_types.h"

namespace mlx::core {

struct complex64_t;
struct complex128_t;

template <typename T>
inline constexpr bool can_convert_to_complex128 =
    !std::is_same_v<T, complex128_t> && std::is_convertible_v<T, double>;

struct complex128_t : public std::complex<double> {
  complex128_t() : std::complex<double>() {};
  complex128_t(double v, double u) : std::complex<double>(v, u) {};
  complex128_t(std::complex<double> v) : std::complex<double>(v) {};

  template <
      typename T,
      typename = typename std::enable_if<can_convert_to_complex128<T>>::type>
  complex128_t(T x) : std::complex<double>(x){};

  operator float() const {
    return real();
  };
};

template <typename T>
inline constexpr bool can_convert_to_complex64 =
    !std::is_same_v<T, complex64_t> && std::is_convertible_v<T, float>;

struct complex64_t : public std::complex<float> {
  complex64_t() : std::complex<float>() {};
  complex64_t(float v, float u) : std::complex<float>(v, u) {};
  complex64_t(std::complex<float> v) : std::complex<float>(v) {};

  template <
      typename T,
      typename = typename std::enable_if<can_convert_to_complex64<T>>::type>
  complex64_t(T x) : std::complex<float>(x){};

  operator float() const {
    return real();
  };
};

inline bool operator>=(const complex64_t& a, const complex64_t& b) {
  return (a.real() > b.real()) ||
      (a.real() == b.real() && a.imag() >= b.imag());
}

inline bool operator>(const complex64_t& a, const complex64_t& b) {
  return (a.real() > b.real()) || (a.real() == b.real() && a.imag() > b.imag());
}

inline complex64_t operator%(complex64_t a, complex64_t b) {
  auto real = a.real() - (b.real() * static_cast<int64_t>(a.real() / b.real()));
  auto imag = a.imag() - (b.imag() * static_cast<int64_t>(a.imag() / b.imag()));
  if (real != 0 && ((real < 0) != (b.real() < 0)))
    real += b.real();
  if (imag != 0 && ((imag < 0) != (b.imag() < 0)))
    imag += b.imag();
  return {real, imag};
}

inline bool operator<=(const complex64_t& a, const complex64_t& b) {
  return operator>=(b, a);
}

inline bool operator<(const complex64_t& a, const complex64_t& b) {
  return operator>(b, a);
}

inline complex64_t operator-(const complex64_t& v) {
  return -static_cast<std::complex<float>>(v);
}

// clang-format off
#define complex_binop_helper(_op_, _operator_, itype)            \
  inline complex64_t _operator_(itype x, const complex64_t& y) { \
    return static_cast<complex64_t>(x) _op_ y;           \
  }                                                              \
  inline complex64_t _operator_(const complex64_t& x, itype y) { \
    return x _op_ static_cast<complex64_t>(y);           \
  }

#define complex_binop(_op_, _operator_)                                               \
  inline complex64_t _operator_(const std::complex<float>& x, const complex64_t& y) { \
    return x _op_ static_cast<std::complex<float>>(y);                                \
  }                                                                                   \
  inline complex64_t _operator_(const complex64_t& x, const std::complex<float>& y) { \
    return static_cast<std::complex<float>>(x) _op_ y;                                \
  }                                                                                   \
  inline complex64_t _operator_(const complex64_t& x, const complex64_t& y) {         \
    return static_cast<std::complex<float>>(x)                                        \
        _op_ static_cast<std::complex<float>>(y);                                     \
  }                                                                                   \
  complex_binop_helper(_op_, _operator_, bool)                                        \
  complex_binop_helper(_op_, _operator_, uint32_t)                                    \
  complex_binop_helper(_op_, _operator_, uint64_t)                                    \
  complex_binop_helper(_op_, _operator_, int32_t)                                     \
  complex_binop_helper(_op_, _operator_, int64_t)                                     \
  complex_binop_helper(_op_, _operator_, float16_t)                                   \
  complex_binop_helper(_op_, _operator_, bfloat16_t)                                  \
  complex_binop_helper(_op_, _operator_, float)
// clang-format on

complex_binop(+, operator+)

} // namespace mlx::core
