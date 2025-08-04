// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

void copy_cpu(const array& src, array& dst, CopyType ctype, Stream stream);
void copy_cpu_inplace(
    const array& src,
    array& dst,
    CopyType ctype,
    Stream stream);

void copy_cpu_inplace(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    Stream stream,
    const std::optional<array>& dynamic_i_offset = std::nullopt,
    const std::optional<array>& dynamic_o_offset = std::nullopt);

// Return a contiguous array with same shape that copies the data of |arr|.
array contiguous_copy_cpu(const array& arr, Stream stream);

} // namespace mlx::core
