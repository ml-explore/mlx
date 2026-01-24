// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, throw an error - scan requires rocPrim integration
  throw std::runtime_error("Scan not yet implemented for ROCm");
}

} // namespace mlx::core
