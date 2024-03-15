// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/common/copy.h"
#include "mlx/stream.h"

namespace mlx::core {

// Generic copy inplace
template <typename stride_t>
void copy_gpu_inplace(
    const array& in,
    array& out,
    const std::vector<stride_t>& strides_inp_pre,
    const std::vector<stride_t>& strides_out_pre,
    int64_t inp_offset,
    int64_t out_offset,
    CopyType ctype,
    const Stream& s);

void copy_gpu(const array& src, array& out, CopyType ctype, const Stream& s);
void copy_gpu(const array& src, array& out, CopyType ctype);

void copy_gpu_inplace(
    const array& src,
    array& out,
    CopyType ctype,
    const Stream& s);

void copy_gpu_inplace(
    const array& in,
    array& out,
    const std::vector<int64_t>& istride,
    int64_t ioffset,
    CopyType ctype,
    const Stream& s);

} // namespace mlx::core
