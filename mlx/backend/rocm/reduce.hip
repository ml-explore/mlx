// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void sum_reduce_kernel(float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Simple reduction placeholder
  if (idx == 0) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      sum += input[i];
    }
    output[0] = sum;
  }
}

void launch_sum_reduce(float* input, float* output, int n, hipStream_t stream) {
  hipLaunchKernelGGL(sum_reduce_kernel, dim3(1), dim3(1), 0, stream, input, output, n);
}

} // namespace mlx::core::rocm 