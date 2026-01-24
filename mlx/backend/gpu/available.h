// Copyright © 2025 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::gpu {

MLX_API bool is_available();

/**
 * Get the number of available GPU devices.
 */
MLX_API int device_count();

/**
 * Get information about a GPU device.
 *
 * Returns a map of device properties. Keys vary by backend:
 *   - device_name (string): Device name
 *   - architecture (string): Architecture identifier
 *   - total_memory/memory_size (size_t): Total device memory
 *   - free_memory (size_t): Available memory (CUDA only)
 *   - uuid (string): Device UUID (CUDA only)
 *   - pci_bus_id (string): PCI bus ID (CUDA only)
 *   - compute_capability_major/minor (size_t): Compute capability (CUDA only)
 */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

} // namespace mlx::core::gpu
