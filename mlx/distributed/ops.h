// Copyright © 2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/distributed/distributed.h"
#include "mlx/utils.h"

namespace mlx::core::distributed {

array all_sum(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice s = {});
array all_gather(
    const array& x,
    std::optional<Group> group = std::nullopt,
    StreamOrDevice S = {});

} // namespace mlx::core::distributed
