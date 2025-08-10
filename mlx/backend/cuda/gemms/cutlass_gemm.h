// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core::cu {

void cutlass_gemm(
    const array& a,
    const array& b,
    array& out,
    int M,
    int N,
    int K,
    cu::CommandEncoder& enc);

}
