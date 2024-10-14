// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_stdlib>
#include "mlx/backend/metal/kernels/steel/utils/type_traits.h"

#pragma METAL internals : enable

namespace mlx {
namespace steel {

///////////////////////////////////////////////////////////////////////////////
// Integral constant with casting
///////////////////////////////////////////////////////////////////////////////

template <typename T, T v>
struct integral_constant {
  typedef T value_type;
  typedef integral_constant type;
  static constant constexpr T value = v;

  METAL_FUNC constexpr operator value_type() const {
    return value;
  }
  METAL_FUNC constexpr value_type operator()() const {
    return value;
  }
};

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <class T>
struct is_integral : bool_constant<metal::is_integral<T>::value> {};

template <class T, T v>
struct is_integral<integral_constant<T, v>>
    : bool_constant<metal::is_integral<T>::value> {};

template <typename T>
constexpr constant bool is_integral_v = is_integral<T>::value;

template <int val>
using Int = integral_constant<int, val>;

template <class T>
struct is_static
    : bool_constant<metal::is_empty<metal::remove_cv_t<T>>::value> {};

///////////////////////////////////////////////////////////////////////////////
// Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

#define integral_const_binop(_op_, _operator_)              \
  template <typename T, T tv, typename U, U uv>             \
  METAL_FUNC constexpr auto _operator_(                     \
      integral_constant<T, tv>, integral_constant<U, uv>) { \
    constexpr auto res = tv _op_ uv;                        \
    return integral_constant<decltype(res), res>{};         \
  }

integral_const_binop(+, operator+);
integral_const_binop(-, operator-);
integral_const_binop(*, operator*);
integral_const_binop(/, operator/);

integral_const_binop(==, operator==);
integral_const_binop(!=, operator!=);
integral_const_binop(<, operator<);
integral_const_binop(>, operator>);
integral_const_binop(<=, operator<=);
integral_const_binop(>=, operator>=);

integral_const_binop(&&, operator&&);
integral_const_binop(||, operator||);

#undef integral_const_binop

///////////////////////////////////////////////////////////////////////////////
// Specail Binary Operators on Integral constants
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    T v,
    typename U,
    typename _E = metal::enable_if<is_integral_v<U> && v == 0>>
METAL_FUNC constexpr integral_constant<decltype(T(0) * U(0)), 0> operator*(
    integral_constant<T, v>,
    U) {
  return {};
}

template <
    typename T,
    T v,
    typename U,
    typename _E = metal::enable_if<is_integral_v<U> && v == 0>>
METAL_FUNC constexpr integral_constant<decltype(T(0) * U(0)), 0> operator*(
    U,
    integral_constant<T, v>) {
  return {};
}

template <
    typename T,
    T v,
    typename U,
    typename _E = metal::enable_if<is_integral_v<U> && v == 0>>
METAL_FUNC constexpr integral_constant<decltype(T(0) / U(1)), 0> operator/(
    integral_constant<T, v>,
    U) {
  return {};
}

template <
    typename T,
    T v,
    typename U,
    typename _E = metal::enable_if<is_integral_v<U> && v == 0>>
METAL_FUNC constexpr integral_constant<decltype(T(0) % U(1)), 0> operator%(
    integral_constant<T, v>,
    U) {
  return {};
}

template <
    typename T,
    T v,
    typename U,
    typename _E = metal::enable_if<is_integral_v<U> && (v == 1 || v == -1)>>
METAL_FUNC constexpr integral_constant<decltype(T(0) % U(1)), 0> operator%(
    U,
    integral_constant<T, v>) {
  return {};
}

///////////////////////////////////////////////////////////////////////////////
// Reduction operators
///////////////////////////////////////////////////////////////////////////////

template <typename T>
METAL_FUNC constexpr T sum(T x) {
  return x;
}

template <typename T, typename... Us>
METAL_FUNC constexpr auto sum(T x, Us... us) {
  return x + sum(us...);
}

} // namespace steel
} // namespace mlx

#pragma METAL internals : disable