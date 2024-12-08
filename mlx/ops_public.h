// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/ops.h"

namespace mlx::core {

template <typename... Args>
inline array std(Args&&... args) {
  return std_dev(std::forward<Args>(args)...);
}
} // namespace mlx::core
