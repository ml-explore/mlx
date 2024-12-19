// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

void slice_gpu(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides,
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
    const std::vector<int>& axes,
    const Shape& low_pad_size,
    const Stream& s);

} // namespace mlx::core
