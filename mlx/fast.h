// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>

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

/** Computes: O = softmax(Q @ K.T) @ V **/
array scaled_dot_product_attention(
    const array& queries,
    const array& keys,
    const array& values,
    const float scale,
    const std::optional<array>& mask = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::fast
