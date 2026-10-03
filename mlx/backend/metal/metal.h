// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/**
 * Set how many operations, and how many MB of distinct input buffers, a Metal
 * command buffer may hold before it is committed (defaults per GPU class, or
 * MLX_MAX_OPS_PER_BUFFER / MLX_MAX_MB_PER_BUFFER at startup). A value <= 0
 * keeps that limit. Returns the previous (max_ops, max_mb). Takes effect for
 * the next encoded operation; weights count toward the MB limit, so decode
 * steps of large models commit far more often than their size suggests.
 */
MLX_API std::pair<int, int> set_command_buffer_limits(int max_ops, int max_mb);

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/* Set a custom path to mlx.metallib. Must be called before any MLX operation.
 */
MLX_API void set_metallib_path(const std::string& path);
MLX_API const std::string& get_metallib_path();

} // namespace mlx::core::metal
