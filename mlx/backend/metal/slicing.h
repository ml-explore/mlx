// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

void slice_gpu(
    const array& in,
    array& out,
    std::vector<int> start_indices,
    std::vector<int> strides,
    const Stream& s);

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s);

} // namespace mlx::core