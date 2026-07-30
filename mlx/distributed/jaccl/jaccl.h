// Copyright © 2025 Apple Inc.

#pragma once

#include <memory>

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed::jaccl {

using GroupImpl = mlx::core::distributed::detail::GroupImpl;

bool is_available();
std::shared_ptr<GroupImpl> init(bool strict = false);
std::shared_ptr<GroupImpl> init(bool strict, AllGatherFactory factory);

} // namespace mlx::core::distributed::jaccl
