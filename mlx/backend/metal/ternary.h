// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

void ternary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s);

void ternary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const char* op,
    const Stream& s);

} // namespace mlx::core
