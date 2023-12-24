// Copyright © 2023 Apple Inc.

#pragma once

#include "array.h"
#include "device.h"
#include "ops.h"
#include "stream.h"

namespace mlx::core::linalg {
array norm(
    const array& a,
    const double ord,
    const std::vector<int>& axis = {},
    bool keepdims = false,
    StreamOrDevice s = {});
array norm(
    const array& a,
    const std::string& ord,
    const std::vector<int>& axis = {},
    bool keepdims = false,
    StreamOrDevice s = {});
array norm(
    const array& a,
    const std::vector<int>& axis = {},
    bool keepdims = false,
    StreamOrDevice s = {});
} // namespace mlx::core::linalg