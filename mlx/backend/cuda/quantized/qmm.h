// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

namespace mlx::core {

void cute_qmm(
    const array& x,
    const array& w,
    const array& scales,
    const array& biases,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder);

} // namespace mlx::core
