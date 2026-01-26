// Copyright © 2025 Apple Inc.

// Note: On Windows, binary_all_windows.cu is used instead (see CMakeLists.txt)
#include <nvtx3/nvtx3.hpp>
#include "mlx/backend/cuda/binary/binary.cuh"

namespace mlx::core {
BINARY_GPU(LessEqual)
} // namespace mlx::core
