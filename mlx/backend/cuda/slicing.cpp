// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/slicing.h"

namespace mlx::core {

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  throw std::runtime_error("concatenate_gpu not implemented in CUDA backend.");
}

} // namespace mlx::core
