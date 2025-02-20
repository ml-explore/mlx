// Copyright © 2025 Apple Inc.

#pragma once

#define MLX_VERSION_MAJOR 0
#define MLX_VERSION_MINOR 24
#define MLX_VERSION_PATCH 2
#define MLX_VERSION_NUMERIC \
  (100000 * MLX_VERSION_MAJOR + 1000 * MLX_VERSION_MINOR + MLX_VERSION_PATCH)

namespace mlx::core {

/* A string representation of the MLX version in the format
 * "major.minor.patch".
 *
 * For dev builds, the version will include the suffix ".devYYYYMMDD+hash"
 */
std::string version();

} // namespace mlx::core
