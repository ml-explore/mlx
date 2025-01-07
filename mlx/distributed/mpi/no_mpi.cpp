// Copyright © 2024 Apple Inc.

#include "mlx/distributed/mpi/mpi.h"

namespace mlx::core::distributed::mpi {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available() {
  return false;
}

std::shared_ptr<GroupImpl> init(bool strict /* = false */) {
  if (strict) {
    throw std::runtime_error("Cannot initialize MPI");
  }
  return nullptr;
}

} // namespace mlx::core::distributed::mpi
