// Copyright © 2025 Apple Inc.

#include <string>

#include "mlx/version.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace mlx::core {

std::string version() {
  return TOSTRING(MLX_VERSION);
}

} // namespace mlx::core
