// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::metal {

struct KernelStats;

/* Check if the Metal backend is available. */
MLX_API bool is_available();

/** Capture a GPU trace, saving it to an absolute file `path` */
MLX_API void start_capture(std::string path = "");
MLX_API void stop_capture();

/** Get information about the GPU and system settings. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info();

/** Kernel-level GPU profiling. */
MLX_API void enable_profiling();
MLX_API void disable_profiling();
MLX_API bool profiling_enabled();
MLX_API std::unordered_map<std::string, KernelStats> get_kernel_stats();
MLX_API void reset_kernel_stats();

} // namespace mlx::core::metal
