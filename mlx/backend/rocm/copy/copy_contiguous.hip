// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/copy/copy.hpp"
#include "mlx/backend/rocm/kernel_utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void copy_contiguous_kernel(
    const char* src,
    char* dst,
    size_t size) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    dst[tid] = src[tid];
  }
}

void copy_contiguous(
    const void* src,
    void* dst,
    size_t size,
    hipStream_t stream) {
  if (size == 0) {
    return;
  }

  const int threads_per_block = 256;
  const int blocks = (size + threads_per_block - 1) / threads_per_block;

  copy_contiguous_kernel<<<blocks, threads_per_block, 0, stream>>>(
      static_cast<const char*>(src),
      static_cast<char*>(dst),
      size);
}

} // namespace mlx::core::rocm 