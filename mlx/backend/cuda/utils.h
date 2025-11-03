// Copyright © 2025 Apple Inc.

// This file include utilities that are used by C++ code (i.e. .cpp files).

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/cuda_utils.h"

namespace mlx::core {

template <typename T>
inline uint max_occupancy_block_dim(T kernel) {
  int _, block_dim;
  if constexpr (std::is_same_v<T, CUfunction>) {
    CHECK_CUDA_ERROR(
        cuOccupancyMaxPotentialBlockSize(&_, &block_dim, kernel, 0, 0, 0));
  } else {
    CHECK_CUDA_ERROR(
        cudaOccupancyMaxPotentialBlockSize(&_, &block_dim, kernel));
  }
  return block_dim;
}

template <typename T>
inline T* gpu_ptr(array& arr) {
  return reinterpret_cast<T*>(
      static_cast<char*>(
          static_cast<cu::CudaBuffer*>(arr.buffer().ptr())->data) +
      arr.offset());
}

template <typename T>
inline const T* gpu_ptr(const array& arr) {
  return gpu_ptr<T>(const_cast<array&>(arr));
}

struct Dtype;

// Convert Dtype to CUDA C++ types.
const char* dtype_to_cuda_type(const Dtype& dtype);

} // namespace mlx::core
