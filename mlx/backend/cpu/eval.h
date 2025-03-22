// Copyright © 2025 Apple Inc.

#pragma once

#include <unordered_set>

#include "mlx/array.h"
#include "mlx/stream.h"

namespace mlx::core::cpu {

void eval(array& arr);
void finalize(
    Stream s,
    std::unordered_set<std::shared_ptr<array::Data>> retain_buffers);

} // namespace mlx::core::cpu
