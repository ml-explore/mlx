// Copyright © 2025 Apple Inc.

#include "mlx/backend/cpu/available.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace mlx::core::cpu {

namespace {

// Get CPU architecture string
std::string get_cpu_architecture() {
#if defined(__aarch64__) || defined(__arm64__)
  return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
  return "x86_64";
#elif defined(__i386__) || defined(__i386) || defined(_M_IX86)
  return "x86";
#elif defined(__arm__) || defined(_M_ARM)
  return "arm";
#else
  return "unknown";
#endif
}

// Get CPU device name
std::string get_cpu_name() {
#ifdef __APPLE__
  char model[256];
  size_t len = sizeof(model);
  if (sysctlbyname("machdep.cpu.brand_string", &model, &len, NULL, 0) == 0) {
    return std::string(model);
  }
#endif
  return get_cpu_architecture();
}

} // anonymous namespace

bool is_available() {
  return true;
}

int device_count() {
  return 1;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int /* device_index */) {
  static auto info =
      std::unordered_map<std::string, std::variant<std::string, size_t>>{
          {"device_name", get_cpu_name()},
          {"architecture", get_cpu_architecture()}};
  return info;
}

} // namespace mlx::core::cpu
