// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <typename T>
__global__ void arange_kernel(T* out, T start, T step, size_t size) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = start + static_cast<T>(idx) * step;
  }
}

} // namespace mlx::core::rocm
