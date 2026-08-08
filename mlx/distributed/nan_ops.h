// Copyright © 2025 Apple Inc.

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "mlx/types/complex.h"
#include "mlx/types/half_types.h"

namespace mlx::core::distributed::detail {

template <typename T>
inline T nan_aware_max(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return std::max(a, b);
  } else if constexpr (std::is_same_v<T, complex64_t>) {
    float r = (std::isnan(a.real()) || std::isnan(b.real()))
        ? static_cast<float>(NAN)
        : (a.real() > b.real() ? a.real() : b.real());
    float i = (std::isnan(a.imag()) || std::isnan(b.imag()))
        ? static_cast<float>(NAN)
        : (a.imag() > b.imag() ? a.imag() : b.imag());
    return complex64_t(r, i);
  } else {
    if (std::isnan(a) || std::isnan(b)) {
      return static_cast<T>(NAN);
    }
    return std::max(a, b);
  }
}

// NaN-aware element-wise minimum. See nan_aware_max.
template <typename T>
inline T nan_aware_min(T a, T b) {
  if constexpr (std::is_integral_v<T>) {
    return std::min(a, b);
  } else if constexpr (std::is_same_v<T, complex64_t>) {
    float r = (std::isnan(a.real()) || std::isnan(b.real()))
        ? static_cast<float>(NAN)
        : (a.real() < b.real() ? a.real() : b.real());
    float i = (std::isnan(a.imag()) || std::isnan(b.imag()))
        ? static_cast<float>(NAN)
        : (a.imag() < b.imag() ? a.imag() : b.imag());
    return complex64_t(r, i);
  } else {
    if (std::isnan(a) || std::isnan(b)) {
      return static_cast<T>(NAN);
    }
    return std::min(a, b);
  }
}

} // namespace mlx::core::distributed::detail
