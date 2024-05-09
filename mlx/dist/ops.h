// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/dist/primitives.h"

namespace mlx::core::dist {

array all_reduce_sum(const array& x, std::shared_ptr<Group> group = nullptr);

} // namespace mlx::core::dist
