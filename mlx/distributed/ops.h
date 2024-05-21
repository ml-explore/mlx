// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/distributed/primitives.h"

namespace mlx::core::distributed {

array all_reduce_sum(const array& x, std::optional<Group> group = std::nullopt);
array all_gather(const array& x, std::optional<Group> group = std::nullopt);

} // namespace mlx::core::distributed
