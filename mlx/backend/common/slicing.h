// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

std::tuple<int64_t, Strides> prepare_slice(
    const array& in,
    const Shape& start_indices,
    const Shape& strides);

void slice(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides);

} // namespace mlx::core
