// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/rocm.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"

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

bool train_arena_begin(size_t capacity_bytes) {
  if (!is_available() || capacity_bytes == 0) {
    return false;
  }
  int dev = 0;
  (void)hipGetDevice(&dev);
  // Use default stream null — arena is stream-agnostic for bump address stability.
  return allocator().train_arena_begin(capacity_bytes, dev, nullptr);
}

void train_arena_reset() {
  allocator().train_arena_reset();
}

void train_arena_end() {
  allocator().train_arena_end();
}

bool train_arena_active() {
  return allocator().train_arena_active();
}

size_t train_arena_high_water() {
  return allocator().train_arena_high_water();
}

bool train_arena_overflowed() {
  return allocator().train_arena_overflowed();
}

} // namespace mlx::core::rocm
