// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core::cu {

void fp_qmv(
    const array& w,
    const array& scales,
    const array& vec,
    array& out,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    CommandEncoder& encoder);

} // namespace mlx::core::cu
