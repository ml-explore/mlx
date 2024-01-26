// Copyright © 2023-2024 Apple Inc.

#include "mlx/primitives.h"

namespace mlx::core {

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::invalid_argument("[Compiled::eval_gpu] NYI");
}

} // namespace mlx::core
