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

} // namespace mlx::core
