// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/backend/metal/device.h"

namespace mlx::core {

void steel_matmul_regular(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    int ldd,
    bool transpose_a,
    bool transpose_b,
    Shape batch_shape,
    Strides batch_strides,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t matrix_stride_out,
    std::vector<array>& copies);

void steel_matmul(
    const Stream& s,
    metal::Device& d,
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    int batch_size_out,
    int lda,
    int ldb,
    bool transpose_a,
    bool transpose_b,
    std::vector<array>& copies,
    Shape batch_shape = {},
    Strides A_batch_stride = {},
    Strides B_batch_stride = {});

} // namespace mlx::core
