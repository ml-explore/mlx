// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/utils.h"

namespace mlx::core::fast {

array rope(
    const array& x,
    int dims,
    bool traditional,
    float base,
    float scale,
    int offset,
    StreamOrDevice s /* = {} */);

} // namespace mlx::core::fast
