// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/backend/common/copy.h"
#include "mlx/stream.h"

#include <optional>

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
    std::optional<array> dynamic_i_offset = std::nullopt,
    std::optional<array> dynamic_o_offset = std::nullopt);

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

// Return a contiguous array with same shape that copies the data of |arr|.
array contiguous_copy_gpu(const array& arr, const Stream& s);

// Copy data from |in| and transpose to |out|'s shape.
void reshape_gpu(const array& in, array& out, Stream s);

// Like the normal ops but safe to call in eval_gpu.
array flatten_in_eval(const array& x, int start_axis, int end_axis, Stream s);
array reshape_in_eval(const array& x, Shape shape, Stream s);
array swapaxes_in_eval(const array& x, int axis1, int axis2);

} // namespace mlx::core
