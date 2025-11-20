// Copyright © 2025 Apple Inc.

#include "mlx/distributed/ibv/ibv.h"

namespace mlx::core::distributed::ibv {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize ibv distributed backend.");
  }
  return nullptr;
}

} // namespace mlx::core::distributed::ibv
