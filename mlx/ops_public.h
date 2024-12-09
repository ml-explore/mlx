// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/ops.h"

namespace mlx::core {

// The "std" function has a name collision with "namespace std" in MSVC after
// "using namespace mlx::core", to work around it in our python bindings code
// we use "std_dev" as name instead and only expose the "std" function in
// public header.
template <typename... Args>
inline array std(Args&&... args) {
  return std_dev(std::forward<Args>(args)...);
}

} // namespace mlx::core
