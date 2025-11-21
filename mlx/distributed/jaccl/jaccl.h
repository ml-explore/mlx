// Copyright © 2025 Apple Inc.

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::jaccl {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available();
std::shared_ptr<GroupImpl> init(bool strict = false);

} // namespace mlx::core::distributed::jaccl
