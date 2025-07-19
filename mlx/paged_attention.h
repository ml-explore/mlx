// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/device.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx::core::paged_attention {

/**
 * \defgroup ops Paged attention operations
 * @{
 */

/** PagedAttention operation. */
array paged_attention(
    const array& q,
    const array& k_cache,
    const array& v_cache,
    const array& block_tables,
    const array& context_lens,
    int max_context_len,
    float softmax_scale,
    std::optional<array> alibi_slopes = std::nullopt,
    std::optional<float> softcapping = std::nullopt,
    StreamOrDevice s_ = {});

/** @} */

} // namespace mlx::core::paged_attention
