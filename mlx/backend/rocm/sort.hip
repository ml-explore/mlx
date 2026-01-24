// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core {

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, throw an error - sorting requires rocThrust integration
  throw std::runtime_error("Sort not yet implemented for ROCm");
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, throw an error
  throw std::runtime_error("ArgSort not yet implemented for ROCm");
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("ArgPartition not yet implemented for ROCm");
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error("Partition not yet implemented for ROCm");
}

} // namespace mlx::core
