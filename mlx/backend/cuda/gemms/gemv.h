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

} // namespace mlx::core::cu
