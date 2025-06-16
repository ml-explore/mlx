// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

#include "mlx/backend/rocm/utils.h"

namespace mlx::core::rocm {

// Basic binary operation kernels will go here
__global__ void add_kernel(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

__global__ void multiply_kernel(float* a, float* b, float* c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

void launch_add(float* a, float* b, float* c, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(add_kernel, dim3(blocks), dim3(threads), 0, stream, a, b, c, n);
}

void launch_multiply(float* a, float* b, float* c, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(multiply_kernel, dim3(blocks), dim3(threads), 0, stream, a, b, c, n);
}

} // namespace mlx::core::rocm 