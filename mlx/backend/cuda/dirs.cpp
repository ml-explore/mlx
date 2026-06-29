// Copyright © 2026 Apple Inc.

namespace mlx::core::cu {

const char* cccl_dir() {
#if defined(MLX_CCCL_DIR)
  return MLX_CCCL_DIR;
#else
  return nullptr;
#endif
}

const char* cuda_bin_dir() {
#if defined(MLX_CUDA_BIN_DIR)
  return MLX_CUDA_BIN_DIR;
#else
  return nullptr;
#endif
}

const char* cudnn_bin_dir() {
#if defined(MLX_CUDNN_BIN_DIR)
  return MLX_CUDNN_BIN_DIR;
#else
  return nullptr;
#endif
}

} // namespace mlx::core::cu
