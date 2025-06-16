// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void relu_kernel(float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

__global__ void sigmoid_kernel(float* input, float* output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

void launch_relu(float* input, float* output, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(relu_kernel, dim3(blocks), dim3(threads), 0, stream, input, output, n);
}

void launch_sigmoid(float* input, float* output, int n, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(sigmoid_kernel, dim3(blocks), dim3(threads), 0, stream, input, output, n);
}

} // namespace mlx::core::rocm 