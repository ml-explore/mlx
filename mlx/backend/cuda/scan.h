// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/primitives.h"
#include "mlx/stream.h"

namespace mlx::core {

void segmented_exclusive_mask_scan_gpu(
    const array& in,
    array& out,
    int64_t segment_size,
    const Stream& s);

} // namespace mlx::core
