// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void layer_norm_kernel(
    float* input, 
    float* output, 
    float* gamma, 
    float* beta, 
    int n, 
    float eps) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (idx < n) {
    // Simplified layer norm placeholder
    // Real implementation would compute mean and variance
    output[idx] = gamma[idx] * input[idx] + beta[idx];
  }
}

void launch_layer_norm(
    float* input, 
    float* output, 
    float* gamma, 
    float* beta, 
    int n, 
    float eps, 
    hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(layer_norm_kernel, dim3(blocks), dim3(threads), 0, stream, 
                     input, output, gamma, beta, n, eps);
}

} // namespace mlx::core::rocm 