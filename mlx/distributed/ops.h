// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/primitives.h"

namespace mlx::core::distributed {

array all_reduce_sum(const array& x, std::shared_ptr<Group> group = nullptr);

} // namespace mlx::core::distributed
