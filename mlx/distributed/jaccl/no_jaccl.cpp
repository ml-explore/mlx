// Copyright © 2025 Apple Inc.

#include "mlx/distributed/jaccl/jaccl.h"

namespace mlx::core::distributed::jaccl {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize jaccl distributed backend.");
  }
  return nullptr;
}

} // namespace mlx::core::distributed::jaccl
