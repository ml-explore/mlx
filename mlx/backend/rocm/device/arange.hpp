// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <typename T>
__global__ void arange_kernel(T* out, T start, T step, size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    out[tid] = start + static_cast<T>(tid) * step;
  }
}

} // namespace mlx::core::rocm