// Copyright © 2024 Apple Inc.

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed {

int Group::rank() {
  return 0;
}

int Group::size() {
  return 1;
}

Group Group::split(int color, int key) {
  throw std::runtime_error("Cannot split the distributed group further");
}

bool is_available() {
  return false;
}

Group init(bool strict /* = false */) {
  return Group(nullptr);
}

} // namespace mlx::core::distributed
