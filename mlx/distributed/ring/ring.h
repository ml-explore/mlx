// Copyright © 2024 Apple Inc.

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::ring {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available();
std::shared_ptr<GroupImpl> init(bool strict = false);

} // namespace mlx::core::distributed::ring
