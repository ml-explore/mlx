// Copyright © 2023-2024 Apple Inc.

#include "mlx/io.h"

namespace mlx::core {

GGUFLoad load_gguf(const std::string&, StreamOrDevice s) {
  throw std::runtime_error(
      "[load_gguf] Compile with MLX_BUILD_GGUF=ON to enable GGUF support.");
}

void save_gguf(
    std::string,
    std::unordered_map<std::string, array>,
    std::unordered_map<std::string, GGUFMetaData>) {
  throw std::runtime_error(
      "[save_gguf] Compile with MLX_BUILD_GGUF=ON to enable GGUF support.");
}

} // namespace mlx::core
