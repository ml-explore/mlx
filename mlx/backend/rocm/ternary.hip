// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void select_kernel(float* condition, float* a, float* b, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = (condition[idx] != 0.0f) ? a[idx] : b[idx];
  }
}

void launch_select(float* condition, float* a, float* b, float* output, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(select_kernel, dim3(blocks), dim3(threads), 0, stream, condition, a, b, output, n);
}

} // namespace mlx::core::rocm 