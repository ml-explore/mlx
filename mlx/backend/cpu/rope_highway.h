// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>
#include <type_traits>

#include "mlx/types/half_types.h"

namespace mlx::core::fast {

constexpr bool has_simd_rope = true;
constexpr int simd_rope_min_size = 1;

enum class RopeHighwayDType : uint8_t {
  Float32,
  Float16,
  BFloat16,
};

template <typename T>
constexpr RopeHighwayDType rope_highway_dtype() {
  if constexpr (std::is_same_v<T, float>) {
    return RopeHighwayDType::Float32;
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return RopeHighwayDType::Float16;
  } else if constexpr (std::is_same_v<T, bfloat16_t>) {
    return RopeHighwayDType::BFloat16;
  } else {
    static_assert(
        std::is_same_v<T, float> || std::is_same_v<T, float16_t> ||
            std::is_same_v<T, bfloat16_t>,
        "Unsupported Highway RoPE dtype");
  }
}

int rope_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims);

int rope_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims);

int rope_non_traditional_highway_forward(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims);

int rope_non_traditional_highway_reverse(
    const void* x_in,
    void* x_out,
    RopeHighwayDType dtype,
    const float* cos_t,
    const float* sin_t,
    int half_dims);

template <typename T, bool forward>
inline int rope_traditional_highway_simd(
    const T* x_in,
    T* x_out,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  if constexpr (forward) {
    return rope_traditional_highway_forward(
        x_in, x_out, rope_highway_dtype<T>(), cos_t, sin_t, half_dims);
  } else {
    return rope_traditional_highway_reverse(
        x_in, x_out, rope_highway_dtype<T>(), cos_t, sin_t, half_dims);
  }
}

template <typename T, bool forward>
inline int rope_non_traditional_highway_simd(
    const T* x_in,
    T* x_out,
    const float* cos_t,
    const float* sin_t,
    int half_dims) {
  if constexpr (forward) {
    return rope_non_traditional_highway_forward(
        x_in, x_out, rope_highway_dtype<T>(), cos_t, sin_t, half_dims);
  } else {
    return rope_non_traditional_highway_reverse(
        x_in, x_out, rope_highway_dtype<T>(), cos_t, sin_t, half_dims);
  }
}

} // namespace mlx::core::fast
