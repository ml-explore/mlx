// Copyright © 2024 Apple Inc.

#pragma once

#include <metal_stdlib>

#pragma METAL internals : enable

namespace metal {

template <typename T>
struct is_empty : metal::bool_constant<__is_empty(T)> {};

#ifdef __cpp_variable_templates
template <typename T>
constexpr constant bool is_empty_v = is_empty<T>::value;
#endif

template <typename... Ts>
struct make_void {
  typedef void type;
};

template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

template <class T, T val>
struct is_integral<integral_constant<T, val>> : true_type {};

template <short val>
struct is_integral<integral_constant<decltype(val), val>> : true_type {};

template <int val>
struct is_integral<integral_constant<decltype(val), val>> : true_type {};

template <long val>
struct is_integral<integral_constant<decltype(val), val>> : true_type {};

template <ushort val>
struct is_integral<integral_constant<decltype(val), val>> : true_type {};

template <uint val>
struct is_integral<integral_constant<decltype(val), val>> : true_type {};

template <size_t val>
struct is_integral<integral_constant<decltype(val), val>> : true_type {};

template <class T>
struct is_static : bool_constant<is_empty<remove_cv_t<T>>::value> {};

} // namespace metal

#pragma METAL internals : disable