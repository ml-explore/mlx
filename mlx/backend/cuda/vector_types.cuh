// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace mlx::core::cu {

// Maps scalar types to their corresponding CUDA vector types
// float -> float2, double -> double2, __half -> __half2, __nv_bfloat16 ->
// __nv_bfloat162
template <typename T>
struct Vector2;

template <>
struct Vector2<double> {
  using type = double2;
};

template <>
struct Vector2<float> {
  using type = float2;
};

template <>
struct Vector2<__half> {
  using type = __half2;
};

template <>
struct Vector2<__nv_bfloat16> {
  using type = __nv_bfloat162;
};

template <typename T>
using Vector2_t = typename Vector2<T>::type;

} // namespace mlx::core::cu