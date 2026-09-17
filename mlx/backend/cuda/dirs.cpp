// Copyright © 2026 Apple Inc.

#include "mlx/backend/common/utils.h"

#include <filesystem>
#include <string>

namespace mlx::core::cu {
namespace {

namespace fs = std::filesystem;

std::string resolve_bin_dir(const char* dir) {
  fs::path path(dir);
  if (path.is_absolute()) {
    return path.string();
  }
  return fs::absolute(current_binary_dir() / path).string();
}

} // namespace

const char* cccl_dir() {
#if defined(MLX_CCCL_DIR)
  return MLX_CCCL_DIR;
#else
  return nullptr;
#endif
}

const char* cuda_bin_dir() {
#if defined(MLX_CUDA_BIN_DIR)
  static const std::string dir = resolve_bin_dir(MLX_CUDA_BIN_DIR);
  return dir.c_str();
#else
  return nullptr;
#endif
}

const char* cudnn_bin_dir() {
#if defined(MLX_CUDNN_BIN_DIR)
  static const std::string dir = resolve_bin_dir(MLX_CUDNN_BIN_DIR);
  return dir.c_str();
#else
  return nullptr;
#endif
}

} // namespace mlx::core::cu
