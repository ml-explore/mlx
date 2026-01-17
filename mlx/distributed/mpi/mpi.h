// Copyright © 2024 Apple Inc.

#include <memory>

#include "mlx/distributed/distributed.h"
#include "mlx/distributed/distributed_impl.h"

namespace mlx::core::distributed::mpi {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available();
std::shared_ptr<GroupImpl> init(bool strict = false);

} // namespace mlx::core::distributed::mpi
