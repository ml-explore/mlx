// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

void slice_gpu(
    const array& in,
    array& out,
    const std::vector<int>& start_indices,
    const std::vector<int>& strides,
    const Stream& s);

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s);

void pad_gpu(
    const array& in,
    const array& val,
    array& out,
    std::vector<int> axes,
    std::vector<int> low_pad_size,
    const Stream& s);

} // namespace mlx::core
