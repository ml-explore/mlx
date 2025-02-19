// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace mlx::core {

std::string version() {
  return std::string(TOSTRING(MLX_VERSION));
}

int version_major() { return MLX_VERSION_MAJOR; }

int version_minor() { return MLX_VERSION_MINOR; }

int version_patch() { return MLX_VERSION_PATCH; }

int version_numeric() { return MLX_VERSION_NUMERIC; }

} // namespace mlx::core
