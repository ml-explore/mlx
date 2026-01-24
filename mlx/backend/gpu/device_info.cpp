// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/available.h"

#if defined(MLX_USE_METAL)
#include "mlx/backend/metal/metal.h"
#elif defined(MLX_USE_CUDA)
#include <cuda_runtime.h>
#include "mlx/backend/cuda/cuda.h"
#endif

namespace mlx::core::gpu {

int device_count() {
#if defined(MLX_USE_METAL)
  return 1;
#elif defined(MLX_USE_CUDA)
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
#else
  return 0;
#endif
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
#if defined(MLX_USE_METAL)
  return metal::device_info();
#elif defined(MLX_USE_CUDA)
  return cu::device_info(device_index);
#else
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;
  return empty;
#endif
}

} // namespace mlx::core::gpu
