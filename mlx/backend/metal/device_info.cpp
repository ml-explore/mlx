// Copyright © 2026 Apple Inc.

#include <sys/sysctl.h>

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/metal/apple_silicon_optimizations.h"
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
    int arch_gen = device.get_architecture_gen();

    size_t memsize = 0;
    size_t length = sizeof(memsize);
    sysctlbyname("hw.memsize", &memsize, &length, NULL, 0);

    size_t rsrc_limit = 0;
    sysctlbyname("iogpu.rsrc_limit", &rsrc_limit, &length, NULL, 0);
    if (rsrc_limit == 0) {
      rsrc_limit = 499000;
    }

    // Detect if this is a Max chip (M1/M2/M3/M4/M5 Max)
    bool is_max_chip = metal::is_max_device(name);
    
    // Detect M5 Max specifically
    bool is_m5_max = metal::is_m5_max(arch_gen, name);
    
    // Get optimal buffer parameters
    auto [max_ops, max_mb] = metal::get_optimal_buffer_params(arch_gen, name);

    // Get additional device capabilities
    size_t memory_clock = 0;  // Would need IOKit to get exact values
    size_t compute_units = 0; // Would need IOKit to get exact values

    return {
        {"device_name", name},
        {"architecture", arch},
        {"architecture_generation", static_cast<size_t>(arch_gen)},
        {"max_buffer_length", raw_device->maxBufferLength()},
        {"max_recommended_working_set_size",
         raw_device->recommendedMaxWorkingSetSize()},
        {"memory_size", memsize},
        {"resource_limit", rsrc_limit},
        {"is_max_chip", is_max_chip},
        {"is_m5_max", is_m5_max},
        {"max_ops_per_buffer", static_cast<size_t>(max_ops)},
        {"max_mb_per_buffer", static_cast<size_t>(max_mb)}};
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
