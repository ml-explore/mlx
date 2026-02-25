// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core::cu {

void fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    array& out,
    int bits,
    int group_size,
    int M,
    int N,
    int K,
    CommandEncoder& encoder,
    Stream s);

} // namespace mlx::core::cu
