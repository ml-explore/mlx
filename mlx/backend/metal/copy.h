// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/backend/common/copy.h"
#include "mlx/stream.h"

namespace mlx::core {

// Generic copy inplace
void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype,
    const Stream& s,
    const std::optional<array>& dynamic_i_offset = std::nullopt,
    const std::optional<array>& dynamic_o_offset = std::nullopt);

void copy_gpu(const array& src, array& out, CopyType ctype, const Stream& s);
void copy_gpu(const array& src, array& out, CopyType ctype);

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s);

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Strides& i_strides,
    int64_t i_offset,
    CopyType ctype,
    const Stream& s);

// Fill the output with the scalar val
void fill_gpu(const array& val, array& out, const Stream& s);

} // namespace mlx::core
