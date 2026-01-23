// Copyright © 2023 Apple Inc.

#pragma once

#include "mlx/api.h"

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

} // namespace mlx::core
