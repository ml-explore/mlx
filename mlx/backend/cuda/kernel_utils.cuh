// Copyright © 2025 Apple Inc.

// This file includes host-only utilies for writing CUDA kernels, the difference
// from backend/cuda/device/utils.cuh is that the latter file only include
// device-only code.

#pragma once

#include <type_traits>

#include "mlx/array.h"
#include "mlx/backend/cuda/device/utils.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <fmt/format.h>
#include <cuda/cmath>

namespace mlx::core {

template <typename F>
void dispatch_1_2_3(int n, F&& f) {
  switch (n) {
    case 1:
      f(std::integral_constant<int, 1>{});
      break;
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 3:
      f(std::integral_constant<int, 3>{});
      break;
  }
}

template <typename F>
void dispatch_bool(bool v, F&& f) {
  if (v) {
    f(std::true_type{});
  } else {
    f(std::false_type{});
  }
}

template <typename F>
void dispatch_block_dim(int threads, F&& f) {
  if (threads <= WARP_SIZE) {
    f(std::integral_constant<int, WARP_SIZE>{});
  } else if (threads <= WARP_SIZE * 2) {
    f(std::integral_constant<int, WARP_SIZE * 2>{});
  } else if (threads <= WARP_SIZE * 4) {
    f(std::integral_constant<int, WARP_SIZE * 4>{});
  } else if (threads <= WARP_SIZE * 8) {
    f(std::integral_constant<int, WARP_SIZE * 8>{});
  } else if (threads <= WARP_SIZE * 16) {
    f(std::integral_constant<int, WARP_SIZE * 16>{});
  } else {
    f(std::integral_constant<int, WARP_SIZE * 32>{});
  }
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
  using type = cu::complex64_t;
};

template <typename T>
using cuda_type_t = typename CTypeToCudaType<T>::type;

// Type traits for detecting floating numbers.
template <typename T>
inline constexpr bool is_floating_v =
    cuda::std::is_same_v<T, float> || cuda::std::is_same_v<T, double> ||
    cuda::std::is_same_v<T, float16_t> || cuda::std::is_same_v<T, bfloat16_t>;

// Type traits for detecting complex numbers.
template <typename T>
inline constexpr bool is_complex_v = cuda::std::is_same_v<T, complex64_t> ||
    cuda::std::is_same_v<T, complex128_t>;

// Type traits for detecting complex or real floating point numbers.
template <typename T>
inline constexpr bool is_inexact_v = is_floating_v<T> || is_complex_v<T>;

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

// Get the num_blocks and block_dims that maximize occupancy for |kernel|,
// assuming each thread handles |work_per_thread| elements of |arr|.
template <typename T>
inline std::tuple<dim3, uint> get_launch_args(
    T kernel,
    size_t size,
    const Shape& shape,
    const Strides& strides,
    bool large,
    int work_per_thread = 1) {
  size_t nthreads = cuda::ceil_div(size, work_per_thread);
  uint block_dim = 1024;
  if (block_dim > nthreads) {
    block_dim = nthreads;
  }
  dim3 num_blocks;
  if (large) {
    num_blocks = get_2d_grid_dims(shape, strides, work_per_thread);
    num_blocks.x = cuda::ceil_div(num_blocks.x, block_dim);
  } else {
    num_blocks.x = cuda::ceil_div(nthreads, block_dim);
  }
  return std::make_tuple(num_blocks, block_dim);
}

template <typename T>
inline std::tuple<dim3, uint> get_launch_args(
    T kernel,
    const array& arr,
    bool large,
    int work_per_thread = 1) {
  return get_launch_args(
      kernel, arr.size(), arr.shape(), arr.strides(), large, work_per_thread);
}

} // namespace mlx::core
