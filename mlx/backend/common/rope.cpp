// Copyright © 2023-2024 Apple Inc.

#include "mlx/fast.h"
#include "mlx/primitives.h"

namespace mlx::core::fast {

void RoPE::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("NYI");
}

} // namespace mlx::core::fast
