// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void random_uniform_kernel(float* output, int n, unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    // Simple LCG placeholder - real implementation would use rocRAND
    unsigned int state = seed + idx;
    state = state * 1103515245 + 12345;
    output[idx] = (float)(state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
  }
}

void launch_random_uniform(float* output, int n, unsigned int seed, hipStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  hipLaunchKernelGGL(random_uniform_kernel, dim3(blocks), dim3(threads), 0, stream, output, n, seed);
}

} // namespace mlx::core::rocm 