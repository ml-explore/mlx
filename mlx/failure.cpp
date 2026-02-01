// Copyright © 2026 Apple Inc.

#include "mlx/backend/gpu/failure.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/failure.h"

namespace mlx::core {

void reset_global_failure() {
  // TODO also reset CPU failure.
  if (gpu::is_available()) {
    gpu::reset_failure();
  }
}

bool global_failure() {
  // TODO also check CPU failure.
  if (gpu::is_available()) {
    return gpu::has_failure();
  }
  return false;
}

} // namespace mlx::core
