// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/distributed/distributed.h"

namespace mlx::core::distributed {

array all_sum(const array& x, std::optional<Group> group = std::nullopt);
array all_gather(const array& x, std::optional<Group> group = std::nullopt);

} // namespace mlx::core::distributed
