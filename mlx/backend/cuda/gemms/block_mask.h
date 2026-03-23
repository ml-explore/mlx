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
    int rows,
    int cols,
    int64_t data_batch_stride,
    int batch_count);

array copy_with_block_mask(
    cu::CommandEncoder& encoder,
    const array& src,
    const array& mask,
    int block_size,
    int rows,
    int cols,
    int batch_count);

} // namespace mlx::core
