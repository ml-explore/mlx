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
  return 1;
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
      rsrc_limit = 499000;
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
