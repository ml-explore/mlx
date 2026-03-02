// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core::cu {

bool can_use_gemv(int M, int N, int K, bool a_transposed, bool b_transposed);

void gemv(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    uint32_t batch_count,
    const mlx::core::Shape& batch_shape,
    const mlx::core::Strides& a_batch_strides,
    const mlx::core::Strides& b_batch_strides,
    CommandEncoder& encoder);

void gather_mv(
    const array& mat,
    const array& vec,
    const array& mat_indices,
    const array& vec_indices,
    array& out,
    int N,
    int K,
    CommandEncoder& encoder);

} // namespace mlx::core::cu
