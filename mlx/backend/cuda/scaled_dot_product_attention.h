// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"

#include <optional>

namespace mlx::core {

bool supports_sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal);

void sdpa_vector(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& sinks,
    Stream s);

} // namespace mlx::core
