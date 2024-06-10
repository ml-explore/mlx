// Copyright © 2023 Apple Inc.

#pragma once

namespace mlx::core {

struct Device {
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

const Device& default_device();

void set_default_device(const Device& d);

bool operator==(const Device& lhs, const Device& rhs);
bool operator!=(const Device& lhs, const Device& rhs);

} // namespace mlx::core
