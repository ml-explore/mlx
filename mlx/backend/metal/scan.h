#pragma once

#include "mlx/array.h"
#include "mlx/primitives.h"

namespace mlx::core {

void scan_gpu_inplace(
    array in,
    array& out,
    Scan::ReduceType reduce_type,
    int axis,
    bool reverse,
    bool inclusive,
    const Stream& s);

} // namespace mlx::core
