// Copyright © 2026 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

#include <optional>

namespace mlx::core {

void qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    array& out,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::CommandEncoder& encoder,
    Stream s);

} // namespace mlx::core
