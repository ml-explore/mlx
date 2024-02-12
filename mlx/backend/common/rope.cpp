// Copyright © 2023-2024 Apple Inc.

#include "mlx/extensions.h"
#include "mlx/primitives.h"

namespace mlx::core::ext {

void RoPE::eval_cpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error("NYI");
}

} // namespace mlx::core::ext
