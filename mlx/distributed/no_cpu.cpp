// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/copy.h"

namespace mlx::core {

void copy(const array&, array&, CopyType) {
  throw std::runtime_error("MLX Compiled without CPU backend.");
}

} // namespace mlx::core
