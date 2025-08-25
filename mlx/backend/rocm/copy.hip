// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void copy_kernel(float* src, float* dst, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

void launch_copy(float* src, float* dst, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(copy_kernel, dim3(blocks), dim3(threads), 0, stream, src, dst, n);
}

} // namespace mlx::core::rocm 