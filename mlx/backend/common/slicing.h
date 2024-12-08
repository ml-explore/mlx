// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

std::tuple<int64_t, Strides> prepare_slice(
    const array& in,
    const Shape& start_indices,
    const Shape& strides);

void shared_buffer_slice(
    const array& in,
    const Strides& out_strides,
    size_t data_offset,
    size_t data_size,
    array& out);

} // namespace mlx::core
