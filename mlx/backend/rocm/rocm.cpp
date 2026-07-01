// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/rocm.h"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

bool is_available() {
  static int available = -1;
  if (available < 0) {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    available = (err == hipSuccess && device_count > 0) ? 1 : 0;
  }
  return available == 1;
}

} // namespace mlx::core::rocm
