// Copyright © 2023 Apple Inc.

#pragma once

#include <variant>

#include "array.h"
#include "device.h"
#include "ops.h"
#include "stream.h"
#include "string.h"

namespace mlx::core::linalg {
array norm(
    const array& a,
    const std::vector<int>& axis = {},
    bool keepdims = false,
    StreamOrDevice s = {});
} // namespace mlx::core::linalg