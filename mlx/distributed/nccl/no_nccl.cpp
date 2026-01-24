// Copyright © 2024 Apple Inc.

#include "mlx/distributed/nccl/nccl.h"

namespace mlx::core::distributed::nccl {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize nccl distributed backend.");
  }
  return nullptr;
}

} // namespace mlx::core::distributed::nccl
