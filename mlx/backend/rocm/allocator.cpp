// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"

namespace mlx::core::rocm {

void* allocate(size_t size) {
  void* ptr;
  check_hip_error("hipMalloc", hipMalloc(&ptr, size));
  return ptr;
}

void deallocate(void* ptr) {
  if (ptr) {
    check_hip_error("hipFree", hipFree(ptr));
  }
}

} // namespace mlx::core::rocm