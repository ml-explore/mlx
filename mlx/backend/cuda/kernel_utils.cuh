// Copyright © 2025 Apple Inc.

// This file includes host-only utilies for writing CUDA kernels, the difference
// from backend/cuda/kernels/utils.cuh is the latter file only include
// device-only code.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/kernels/utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda/std/type_traits>

namespace mlx::core {

// Helper macros for dispatch macros (see below).
#define MLX_INTERNAL_IF_CASE(DIM, BLOCK_DIM, ...) \
  }                                               \
  else if (_num_threads <= DIM) {                 \
    constexpr uint32_t BLOCK_DIM = DIM;           \
    __VA_ARGS__;

#define MLX_INTERNAL_IF_CASE_DIMS(NUM_THREADS, BLOCK_DIM, ...) \
  {                                                            \
    uint32_t _num_threads = NUM_THREADS;                       \
    if (false) {                                               \
      MLX_INTERNAL_IF_CASE(32, BLOCK_DIM, __VA_ARGS__)         \
      MLX_INTERNAL_IF_CASE(64, BLOCK_DIM, __VA_ARGS__)         \
      MLX_INTERNAL_IF_CASE(128, BLOCK_DIM, __VA_ARGS__)        \
      MLX_INTERNAL_IF_CASE(256, BLOCK_DIM, __VA_ARGS__)        \
      MLX_INTERNAL_IF_CASE(512, BLOCK_DIM, __VA_ARGS__)        \
    } else {                                                   \
      constexpr uint32_t BLOCK_DIM = 1024;                     \
      __VA_ARGS__;                                             \
    }                                                          \
  }

// Some kernels use CUB which requires block_dim to be known at compile-time,
// use this macro to dispatch constexpr block_dim for the num_threads.
#define MLX_SWITCH_BLOCK_DIM(NUM_THREADS, BLOCK_DIM, ...) \
  MLX_INTERNAL_IF_CASE_DIMS(NUM_THREADS, BLOCK_DIM, __VA_ARGS__)

// Maps CPU types to CUDA types.
template <typename T>
struct CTypeToCudaType {
  using type = T;
};

template <>
struct CTypeToCudaType<float16_t> {
  using type = __half;
};

template <>
struct CTypeToCudaType<bfloat16_t> {
  using type = __nv_bfloat16;
};

template <>
struct CTypeToCudaType<complex64_t> {
  using type = cuComplex;
};

template <typename T>
using cuda_type_t = typename CTypeToCudaType<T>::type;

// Type traits for detecting floating numbers.
template <typename T>
inline constexpr bool is_floating_v =
    cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, double> ||
    cuda::std::is_same_v<T, float16_t> || cuda::std::is_same_v<T, bfloat16_t>;

// Utility to copy data from vector to array in host.
template <typename T>
inline cuda::std::array<T, MAX_NDIM> const_param(const std::vector<T>& vec) {
  if (vec.size() > MAX_NDIM) {
    throw std::runtime_error("ndim can not be larger than 8.");
  }
  cuda::std::array<T, MAX_NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

} // namespace mlx::core
