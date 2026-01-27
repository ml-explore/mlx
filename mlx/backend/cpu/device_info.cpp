// Copyright © 2026 Apple Inc.

#include "mlx/backend/cpu/device_info.h"

#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/utsname.h>
#elif defined(_WIN32)
#include <windows.h>
#else
#include <sys/utsname.h>
#include <fstream>
#endif

namespace mlx::core::cpu {

namespace {

// Get CPU architecture string at runtime
std::string get_cpu_architecture() {
#ifdef _WIN32
  // Use GetNativeSystemInfo to get the actual hardware architecture,
  // even when running under WoW64 emulation
  SYSTEM_INFO sysInfo;
  GetNativeSystemInfo(&sysInfo);
  switch (sysInfo.wProcessorArchitecture) {
    case PROCESSOR_ARCHITECTURE_AMD64:
      return "x86_64";
    case PROCESSOR_ARCHITECTURE_ARM64:
      return "arm64";
    case PROCESSOR_ARCHITECTURE_INTEL:
      return "x86";
    case PROCESSOR_ARCHITECTURE_ARM:
      return "arm";
    default:
      return "unknown";
  }
#else
  // Use uname() for runtime detection on Unix-like systems.
  // This returns the actual hardware architecture (e.g., "arm64" on Apple
  // Silicon even when running x86_64 binaries via Rosetta 2)
  struct utsname info;
  if (uname(&info) == 0) {
    return std::string(info.machine);
  }
  return "unknown";
#endif
}

// Get CPU device name (brand string)
std::string get_cpu_name() {
#ifdef __APPLE__
  char model[256];
  size_t len = sizeof(model);
  if (sysctlbyname("machdep.cpu.brand_string", &model, &len, NULL, 0) == 0) {
    return std::string(model);
  }
#elif defined(_WIN32)
  // Read CPU brand string from registry
  HKEY hKey;
  if (RegOpenKeyExA(
          HKEY_LOCAL_MACHINE,
          "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
          0,
          KEY_READ,
          &hKey) == ERROR_SUCCESS) {
    char brand[256];
    DWORD size = sizeof(brand);
    if (RegQueryValueExA(
            hKey, "ProcessorNameString", NULL, NULL, (LPBYTE)brand, &size) ==
        ERROR_SUCCESS) {
      RegCloseKey(hKey);
      return std::string(brand);
    }
    RegCloseKey(hKey);
  }
#else
  // Try reading from /proc/cpuinfo on Linux
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    std::string line;
    while (std::getline(cpuinfo, line)) {
      if (line.starts_with("model name")) {
        if (auto n = line.find(": "); n != std::string::npos) {
          return line.substr(n + 2);
        }
      }
    }
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
