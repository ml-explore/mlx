// Copyright © 2025 Apple Inc.

// This file includes host-only utilities for writing HIP kernels, the
// difference from backend/rocm/device/utils.hpp is that the latter file only
// include device-only code.

#pragma once

#include <type_traits>

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device/config.h"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <sstream>
#include <stdexcept>

namespace mlx::core {

// Get GPU pointer from array without synchronization.
// This should be used when passing pointers to GPU kernels.
// For CPU access to managed memory, use array::data<T>() which synchronizes.
template <typename T>
inline T* gpu_ptr(array& arr) {
  return reinterpret_cast<T*>(
      static_cast<char*>(
          static_cast<rocm::RocmBuffer*>(arr.buffer().ptr())->data) +
      arr.offset());
}

// For const array, keep constness in pointer unless it is untyped.
template <typename T>
inline std::conditional_t<std::is_same_v<T, void>, void*, const T*> gpu_ptr(
    const array& arr) {
  return gpu_ptr<T>(const_cast<array&>(arr));
}

// Note: WARP_SIZE and MAX_NDIM are defined in device/config.h

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

// Maps CPU types to HIP types.
template <typename T>
struct CTypeToHipType {
  using type = T;
};

template <>
struct CTypeToHipType<float16_t> {
  using type = __half;
};

template <>
struct CTypeToHipType<bfloat16_t> {
  using type = hip_bfloat16;
};

template <>
struct CTypeToHipType<complex64_t> {
  using type = hipFloatComplex;
};

template <typename T>
using hip_type_t = typename CTypeToHipType<T>::type;

// Type traits for detecting floating numbers.
template <typename T>
inline constexpr bool is_floating_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, float16_t> || std::is_same_v<T, bfloat16_t>;

// Type traits for detecting complex numbers.
template <typename T>
inline constexpr bool is_complex_v =
    std::is_same_v<T, complex64_t> || std::is_same_v<T, complex128_t> ||
    std::is_same_v<T, hipFloatComplex> || std::is_same_v<T, hipDoubleComplex>;

// Type traits for detecting complex or real floating point numbers.
template <typename T>
inline constexpr bool is_inexact_v = is_floating_v<T> || is_complex_v<T>;

// Utility to copy data from vector to array in host.
template <int NDIM = MAX_NDIM, typename T = int32_t>
inline rocm::hip_array<T, NDIM> const_param(const SmallVector<T>& vec) {
  if (vec.size() > NDIM) {
    std::ostringstream oss;
    oss << "ndim can not be larger than " << NDIM << ".";
    throw std::runtime_error(oss.str());
  }
  rocm::hip_array<T, NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.data_);
  return result;
}

// Overload for std::vector
template <int NDIM = MAX_NDIM, typename T = int32_t>
inline rocm::hip_array<T, NDIM> const_param(const std::vector<T>& vec) {
  if (vec.size() > NDIM) {
    std::ostringstream oss;
    oss << "ndim can not be larger than " << NDIM << ".";
    throw std::runtime_error(oss.str());
  }
  rocm::hip_array<T, NDIM> result;
  std::copy_n(vec.begin(), vec.size(), result.data_);
  return result;
}

// Compute the grid and block dimensions
inline dim3 get_block_dims(int dim0, int dim1, int dim2, int pow2 = 10) {
  int block_x = 1;
  int block_y = 1;
  int block_z = 1;

  // Try to maximize occupancy while respecting dimension sizes
  int total_threads = 1 << pow2; // Default to 1024 threads

  // Distribute threads across dimensions
  while (block_x < dim0 && block_x < 32) {
    block_x *= 2;
  }
  while (block_y < dim1 && block_x * block_y < total_threads) {
    block_y *= 2;
  }
  while (block_z < dim2 && block_x * block_y * block_z < total_threads) {
    block_z *= 2;
  }

  return dim3(block_x, block_y, block_z);
}

inline dim3 get_2d_grid_dims(const Shape& shape, const Strides& strides) {
  Dims dims = get_2d_grid_dims_common(shape, strides);
  return dim3(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}

inline dim3
get_2d_grid_dims(const Shape& shape, const Strides& strides, size_t divisor) {
  // Compute the 2d grid dimensions such that the total size of the grid is
  // divided by divisor.
  size_t grid_x = 1;
  size_t grid_y = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (strides[i] == 0) {
      continue;
    }

    // No need to add this shape we can just remove it from the divisor.
    if (divisor % shape[i] == 0) {
      divisor /= shape[i];
      continue;
    }

    if (grid_x * shape[i] < UINT32_MAX) {
      grid_x *= shape[i];
    } else {
      grid_y *= shape[i];
    }
  }
  if (grid_y > UINT32_MAX || grid_x > UINT32_MAX) {
    throw std::runtime_error("Unable to safely factor shape.");
  }
  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }
  return dim3(static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1);
}

inline std::pair<dim3, dim3> get_grid_and_block(int dim0, int dim1, int dim2) {
  auto block_dims = get_block_dims(dim0, dim1, dim2);
  dim3 grid_dims(
      (dim0 + block_dims.x - 1) / block_dims.x,
      (dim1 + block_dims.y - 1) / block_dims.y,
      (dim2 + block_dims.z - 1) / block_dims.z);
  return {grid_dims, block_dims};
}

// Get the num_blocks and block_dims for a kernel
inline std::tuple<dim3, uint> get_launch_args(
    size_t size,
    const Shape& shape,
    const Strides& strides,
    bool large,
    int work_per_thread = 1) {
  size_t adjusted_size = (size + work_per_thread - 1) / work_per_thread;
  int block_size = 256;
  int num_blocks = (adjusted_size + block_size - 1) / block_size;
  num_blocks = std::min(num_blocks, 65535);
  return {dim3(num_blocks), block_size};
}

inline std::tuple<dim3, uint>
get_launch_args(const array& arr, bool large, int work_per_thread = 1) {
  return get_launch_args(
      arr.size(), arr.shape(), arr.strides(), large, work_per_thread);
}

// Ceil division utility
template <typename T>
inline T ceildiv(T a, T b) {
  return (a + b - 1) / b;
}

} // namespace mlx::core
