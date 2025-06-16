// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void softmax_kernel(float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < n) {
    // Simplified softmax placeholder - real implementation needs reduction
    output[idx] = expf(input[idx]);
  }
}

void launch_softmax(float* input, float* output, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(softmax_kernel, dim3(blocks), dim3(threads), 0, stream, input, output, n);
}

} // namespace mlx::core::rocm 