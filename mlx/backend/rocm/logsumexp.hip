// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

__global__ void logsumexp_kernel(float* input, float* output, int n) {
  // Placeholder implementation
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  (void)input; (void)output; (void)n; (void)idx;
}

} // namespace mlx::core::rocm 