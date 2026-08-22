// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/**
 * Select canonical reduction plans for supported short-block Metal inference
 * kernels up to `limit`. This covers dense matmul and addmm,
 * transposed-weight quantized matmul, and the vector SDPA path (whose
 * query-length cap is eight). It does not make every MLX operation batch
 * invariant. Set to zero to disable.
 *
 * The setting is process-wide and should remain stable while concurrent work
 * is executing. Increasing the limit can reduce performance. The default is
 * controlled by the ``MLX_METAL_BATCH_INVARIANT_LIMIT`` environment variable
 * and is zero when the variable is unset.
 */
MLX_API void set_batch_invariant_limit(int limit);
MLX_API int get_batch_invariant_limit();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

} // namespace mlx::core::metal
