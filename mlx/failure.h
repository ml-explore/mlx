// Copyright © 2026 Apple Inc.

#pragma once

#include <cstdint>

namespace mlx::core {

enum class FailureCode : int32_t {
  NoFailure = -1,
  BoundsFailure,
};

void reset_global_failure();

bool global_failure();

} // namespace mlx::core
