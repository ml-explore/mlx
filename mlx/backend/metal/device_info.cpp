// Copyright © 2026 Apple Inc.

#include <sys/sysctl.h>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"

namespace mlx::core::gpu {

bool is_available() {
  return metal::is_available();
}

int device_count() {
  try {
    metal::device(Device::gpu);
    return 1;
  } catch (...) {
    return 0;
  }
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  auto init_device_info = []()
      -> std::unordered_map<std::string, std::variant<std::string, size_t>> {
    auto pool = metal::new_scoped_memory_pool();
    auto& device = metal::device(mlx::core::Device::gpu);
    auto raw_device = device.mtl_device();
    auto name = std::string(raw_device->name()->utf8String());
    auto arch = device.get_architecture();

    size_t memsize = 0;
    size_t length = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &length, NULL, 0);

    size_t rsrc_limit = 0;
    sysctlbyname("iogpu.rsrc_limit", &rsrc_limit, &length, NULL, 0);
    if (rsrc_limit == 0) {
      // Default macOS rsrc_limit (499K) is too tight for large unified
      // memory training. Scale with hw.memsize so small machines stay
      // on stock and only large-RAM devices get the uplift.
      constexpr size_t kBaseRsrcLimit = 499000;
      constexpr size_t kGB = 1024ULL * 1024ULL * 1024ULL;
      if (memsize >= 384 * kGB) {
        rsrc_limit = kBaseRsrcLimit * 3;       // M3 Ultra 512 GB
      } else if (memsize >= 64 * kGB) {
        rsrc_limit = kBaseRsrcLimit * 2;       // 64-128 GB Max/Studio
      } else if (memsize >= 24 * kGB) {
        rsrc_limit = (kBaseRsrcLimit * 3) / 2; // 24-36 GB Pro (1.5x)
      } else {
        rsrc_limit = kBaseRsrcLimit;           // < 24 GB: stock
      }
    }

    return {
        {"device_name", name},
        {"architecture", arch},
        {"max_buffer_length", raw_device->maxBufferLength()},
        {"max_recommended_working_set_size",
         raw_device->recommendedMaxWorkingSetSize()},
        {"memory_size", memsize},
        {"resource_limit", rsrc_limit}};
  };
  static auto device_info_ = init_device_info();
  static std::unordered_map<std::string, std::variant<std::string, size_t>>
      empty;

  if (device_index == 0) {
    return device_info_;
  } else {
    return empty;
  }
}

} // namespace mlx::core::gpu
