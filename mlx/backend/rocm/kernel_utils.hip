// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// Utility functions for HIP kernels

__device__ inline int get_global_id() {
  return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__ inline int get_local_id() {
  return threadIdx.x;
}

__device__ inline int get_group_id() {
  return blockIdx.x;
}

__device__ inline int get_local_size() {
  return blockDim.x;
}

__device__ inline int get_num_groups() {
  return gridDim.x;
}

} // namespace mlx::core::rocm 