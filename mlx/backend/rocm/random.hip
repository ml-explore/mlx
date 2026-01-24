// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/random.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>

namespace mlx::core {

namespace rocm {

template <typename T>
__global__ void random_uniform_kernel(
    T* out,
    size_t size,
    T low,
    T high,
    unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  
  hiprandState state;
  hiprand_init(seed, idx, 0, &state);
  
  float r = hiprand_uniform(&state);
  out[idx] = static_cast<T>(low + r * (high - low));
}

template <typename T>
__global__ void random_normal_kernel(
    T* out,
    size_t size,
    T mean,
    T stddev,
    unsigned long long seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) return;
  
  hiprandState state;
  hiprand_init(seed, idx, 0, &state);
  
  float r = hiprand_normal(&state);
  out[idx] = static_cast<T>(mean + r * stddev);
}

} // namespace rocm

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);
  
  out.set_data(allocator::malloc(out.nbytes()));
  
  // For now, use a simple random implementation
  // TODO: Implement proper random bits generation
  throw std::runtime_error("RandomBits not yet fully implemented for ROCm");
}

} // namespace mlx::core
