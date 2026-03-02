// Copyright © 2023-2025 Apple Inc.

#pragma once

#include "mlx/api.h"

#include <string>
#include <unordered_map>
#include <variant>

namespace mlx::core {

struct MLX_API Device {
  enum class DeviceType {
    cpu,
    gpu,
  };

  static constexpr DeviceType cpu = DeviceType::cpu;
  static constexpr DeviceType gpu = DeviceType::gpu;

  Device(DeviceType type, int index = 0) : type(type), index(index) {}

  DeviceType type;
  int index;
};

MLX_API const Device& default_device();

MLX_API void set_default_device(const Device& d);

MLX_API bool operator==(const Device& lhs, const Device& rhs);
MLX_API bool operator!=(const Device& lhs, const Device& rhs);

MLX_API bool is_available(const Device& d);

/** Get the number of available devices for the given device type. */
MLX_API int device_count(Device::DeviceType type);

/**
 * Get information about a device.
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
    device_info(const Device& d = default_device());

} // namespace mlx::core
