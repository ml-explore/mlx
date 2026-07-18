// Copyright © 2026 Apple Inc.

#pragma once

#include <optional>

#include "mlx/api.h"
#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::random {

/** Internal search of a canonical uint64 CDF using uint32 random words. */
MLX_API array categorical_search(
    const array& cdf,
    const array& random_bits,
    StreamOrDevice s = {});

/** Internal fixed-point inverse-CDF categorical implementation. */
MLX_API array categorical_fixed(
    const array& logits,
    int axis,
    int num_samples,
    const std::optional<array>& key = std::nullopt,
    StreamOrDevice s = {});

} // namespace mlx::core::random
