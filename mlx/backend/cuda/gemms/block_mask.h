// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core {

void apply_block_mask(
    cu::CommandEncoder& encoder,
    array& data,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    int64_t batch_count);

array copy_with_block_mask(
    cu::CommandEncoder& encoder,
    const array& src,
    const array& mask,
    int block_size,
    int64_t rows,
    int64_t cols,
    int64_t batch_count);

} // namespace mlx::core
