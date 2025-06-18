// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>
#include <array>

namespace mlx::core::rocm {

// Constants
constexpr int MAX_DIMS = 8;

// HIP array type for passing arrays to kernels
template <typename T, int N>
using hip_array = std::array<T, N>;

// Helper to create hip_array from vector
template <int N, typename T>
__host__ hip_array<T, N> make_hip_array(const std::vector<T>& vec) {
  hip_array<T, N> arr;
  for (int i = 0; i < N && i < vec.size(); ++i) {
    arr[i] = vec[i];
  }
  return arr;
}

template <typename T>
__host__ hip_array<T, MAX_DIMS> make_hip_array(const std::vector<T>& vec) {
  return make_hip_array<MAX_DIMS>(vec);
}

// Type mapping from MLX types to HIP types
template <typename T>
using hip_type_t = T;

template <>
using hip_type_t<float16> = __half;

template <>
using hip_type_t<bfloat16> = __hip_bfloat16;

template <>
using hip_type_t<complex64> = hipFloatComplex;

// Element to location mapping for general broadcasting
template <int NDIM>
__device__ std::pair<int64_t, int64_t> elem_to_loc_nd(
    int64_t elem,
    const int32_t* shape,
    const int64_t* a_strides,
    const int64_t* b_strides) {
  int64_t a_idx = 0;
  int64_t b_idx = 0;

  for (int i = NDIM - 1; i >= 0; --i) {
    int64_t pos_in_dim = elem % shape[i];
    elem /= shape[i];
    a_idx += pos_in_dim * a_strides[i];
    b_idx += pos_in_dim * b_strides[i];
  }

  return {a_idx, b_idx};
}

// 4D specialization for performance
__device__ inline std::pair<int64_t, int64_t> elem_to_loc_4d(
    int64_t elem,
    const int32_t* shape,
    const int64_t* a_strides,
    const int64_t* b_strides,
    int ndim) {
  int64_t a_idx = 0;
  int64_t b_idx = 0;

  for (int i = ndim - 1; i >= 0; --i) {
    int64_t pos_in_dim = elem % shape[i];
    elem /= shape[i];
    a_idx += pos_in_dim * a_strides[i];
    b_idx += pos_in_dim * b_strides[i];
  }

  return {a_idx, b_idx};
}

// Launch configuration calculation
template <typename Kernel>
std::pair<dim3, dim3>
get_launch_args(Kernel kernel, const array& out, bool large = false) {
  int threads_per_block = 256;
  int64_t total_threads = out.size();

  if (large) {
    // For large arrays, use more blocks
    int64_t blocks =
        (total_threads + threads_per_block - 1) / threads_per_block;
    return {dim3(blocks), dim3(threads_per_block)};
  } else {
    int blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    return {dim3(blocks), dim3(threads_per_block)};
  }
}

template <typename Kernel>
std::pair<dim3, dim3> get_launch_args(
    Kernel kernel,
    int64_t size,
    const std::vector<int>& shape,
    const std::vector<size_t>& strides,
    bool large = false) {
  int threads_per_block = 256;

  if (large) {
    int64_t blocks = (size + threads_per_block - 1) / threads_per_block;
    return {dim3(blocks), dim3(threads_per_block)};
  } else {
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    return {dim3(blocks), dim3(threads_per_block)};
  }
}

// Cooperative groups thread rank equivalent
namespace cooperative_groups {
class grid_group {
 public:
  __device__ int64_t thread_rank() const {
    return blockIdx.x * blockDim.x + threadIdx.x;
  }
};

__device__ grid_group this_grid() {
  return grid_group{};
}
} // namespace cooperative_groups

} // namespace mlx::core::rocm