#pragma once

#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx::core {

void scan_gpu(
    array in,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    const Stream& s,
    bool allow_in_buffer_donation = true);

} // namespace mlx::core
