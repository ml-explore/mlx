// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/device.h"

namespace mlx::core {

void gemv(
    rocm::CommandEncoder& encoder,
    bool transpose_a,
    int M,
    int N,
    float alpha,
    const array& a,
    int lda,
    const array& x,
    float beta,
    array& y,
    Dtype dtype);

bool can_use_gemv(int M, int N, int K, bool trans_a, bool trans_b);

void gather_mv(
    const array& mat,
    const array& vec,
    const array& mat_indices,
    const array& vec_indices,
    array& out,
    int M,
    int K,
    rocm::CommandEncoder& encoder);

} // namespace mlx::core
