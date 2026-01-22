// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/api.h"

namespace mlx::core::cu {

/* Check if the CUDA backend is available. */
MLX_API bool is_available();

} // namespace mlx::core::cu
