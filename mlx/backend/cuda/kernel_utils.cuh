// Copyright © 2025 Apple Inc.

// This file includes host-only utilies for writing CUDA kernels, the difference
// from backend/cuda/device/utils.cuh is that the latter file only include
// device-only code.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cuComplex.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <fmt/format.h>
#include <cuda/cmath>

namespace mlx::core {

// Convert a number between 1~3 to constexpr.
#define MLX_SWITCH_1_2_3(N, NDIM, ...) \
  switch (N) {                         \
    case 1: {                          \
      constexpr int NDIM = 1;          \
      __VA_ARGS__;                     \
      break;                           \
    }                                  \
    case 2: {                          \
      constexpr int NDIM = 2;          \
      __VA_ARGS__;                     \
      break;                           \
    }                                  \
    case 3: {                          \
      constexpr int NDIM = 3;          \
      __VA_ARGS__;                     \
      break;                           \
    }                                  \
  }

// Like MLX_SWITCH_ALL_TYPES but for booleans.
#define MLX_SWITCH_BOOL(BOOL, BOOL_ALIAS, ...) \
  if (BOOL) {                                  \
    constexpr bool BOOL_ALIAS = true;          \
    __VA_ARGS__;                               \
  } else {                                     \
    constexpr bool BOOL_ALIAS = false;         \
    __VA_ARGS__;                               \
  }

// Convert a block_dim to constexpr between WARP_SIZE and WARP_SIZE ^ 2.
#define MLX_SWITCH_BLOCK_DIM(NUM_THREADS, BLOCK_DIM, ...)   \
  {                                                         \
    uint32_t _num_threads = NUM_THREADS;                    \
    if (_num_threads <= WARP_SIZE) {                        \
      constexpr uint32_t BLOCK_DIM = WARP_SIZE;             \
      __VA_ARGS__;                                          \
    } else if (_num_threads <= WARP_SIZE * 2) {             \
      constexpr uint32_t BLOCK_DIM = WARP_SIZE * 2;         \
      __VA_ARGS__;                                          \
    } else if (_num_threads <= WARP_SIZE * 4) {             \
      constexpr uint32_t BLOCK_DIM = WARP_SIZE * 4;         \
      __VA_ARGS__;                                          \
    } else if (_num_threads <= WARP_SIZE * 8) {             \
      constexpr uint32_t BLOCK_DIM = WARP_SIZE * 8;         \
      __VA_ARGS__;                                          \
    } else if (_num_threads <= WARP_SIZE * 16) {            \
      constexpr uint32_t BLOCK_DIM = WARP_SIZE * 16;        \
      __VA_ARGS__;                                          \
    } else {                                                \
      constexpr uint32_t BLOCK_DIM = WARP_SIZE * WARP_SIZE; \
      __VA_ARGS__;                                          \
    }                                                       \
  }

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
template <int NDIM = MAX_NDIM, typename T = int32_t>
inline cuda::std::array<T, NDIM> const_param(const std::vector<T>& vec) {
  if (vec.size() > NDIM) {
    throw std::runtime_error(
        fmt::format("ndim can not be larger than {}.", NDIM));
  }
  cuda::std::array<T, NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.begin());
  return result;
}

// Compute the grid and block dimensions, check backend/common/utils.h for docs.
dim3 get_block_dims(int dim0, int dim1, int dim2, int pow2 = 10);
dim3 get_2d_grid_dims(const Shape& shape, const Strides& strides);
dim3 get_2d_grid_dims(
    const Shape& shape,
    const Strides& strides,
    size_t divisor);
std::pair<dim3, dim3> get_grid_and_block(int dim0, int dim1, int dim2);

// Return a block size that achieves maximum potential occupancy for kernel.
template <typename T>
inline uint max_occupancy_block_dim(T kernel) {
  int _, block_dim;
  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&_, &block_dim, kernel));
  return block_dim;
}

// Get the num_blocks and block_dims that maximize occupancy for |kernel|,
// assuming each thread handles |work_per_thread| elements of |arr|.
template <typename T>
inline std::tuple<dim3, uint> get_launch_args(
    T kernel,
    const array& arr,
    bool large,
    int work_per_thread = 1) {
  size_t nthreads = cuda::ceil_div(arr.size(), work_per_thread);
  uint block_dim = max_occupancy_block_dim(kernel);
  if (block_dim > nthreads) {
    block_dim = nthreads;
  }
  dim3 num_blocks;
  if (large) {
    num_blocks = get_2d_grid_dims(arr.shape(), arr.strides(), work_per_thread);
    num_blocks.x = cuda::ceil_div(num_blocks.x, block_dim);
  } else {
    num_blocks.x = cuda::ceil_div(nthreads, block_dim);
  }
  return std::make_tuple(num_blocks, block_dim);
}

} // namespace mlx::core
