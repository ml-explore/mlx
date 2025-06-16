// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void argmax_kernel(float* input, int* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // Simple argmax placeholder
  if (idx == 0) {
    int max_idx = 0;
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
      if (input[i] > max_val) {
        max_val = input[i];
        max_idx = i;
      }
    }
    output[0] = max_idx;
  }
}

void launch_argmax(float* input, int* output, int n, hipStream_t stream) {
  hipLaunchKernelGGL(argmax_kernel, dim3(1), dim3(1), 0, stream, input, output, n);
}

} // namespace mlx::core::rocm 