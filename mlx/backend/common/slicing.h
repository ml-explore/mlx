// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

std::tuple<bool, int64_t, std::vector<int64_t>> prepare_slice(
    const array& in,
    std::vector<int>& start_indices,
    std::vector<int>& strides);

void shared_buffer_slice(
    const array& in,
    const std::vector<size_t>& out_strides,
    size_t data_offset,
    array& out);

} // namespace mlx::core
