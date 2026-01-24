// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  // LogSumExp = log(sum(exp(x - max(x)))) + max(x)
  // For now, throw an error - this requires a specialized kernel
  throw std::runtime_error("LogSumExp not yet implemented for ROCm");
}

} // namespace mlx::core
