// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

void transpose(const array& in, array& out, const std::vector<int>& axes);

} // namespace mlx::core
