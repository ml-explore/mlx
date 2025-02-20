// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace mlx::core {

constexpr const char* version() {
  return TOSTRING(MLX_VERSION);
}

constexpr int version_major() {
  return MLX_VERSION_MAJOR;
}

constexpr int version_minor() {
  return MLX_VERSION_MINOR;
}

constexpr int version_patch() {
  return MLX_VERSION_PATCH;
}

constexpr int version_numeric() {
  return MLX_VERSION_NUMERIC;
}

} // namespace mlx::core
