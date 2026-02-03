// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/rocm/utils.h"

#include <hip/hip_runtime.h>

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mlx::core {

namespace {

std::string format_uuid(const hipUUID& uuid) {
  char buf[64];
  snprintf(
      buf,
      sizeof(buf),
      "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      (unsigned char)uuid.bytes[0],
      (unsigned char)uuid.bytes[1],
      (unsigned char)uuid.bytes[2],
      (unsigned char)uuid.bytes[3],
      (unsigned char)uuid.bytes[4],
      (unsigned char)uuid.bytes[5],
      (unsigned char)uuid.bytes[6],
      (unsigned char)uuid.bytes[7],
      (unsigned char)uuid.bytes[8],
      (unsigned char)uuid.bytes[9],
      (unsigned char)uuid.bytes[10],
      (unsigned char)uuid.bytes[11],
      (unsigned char)uuid.bytes[12],
      (unsigned char)uuid.bytes[13],
      (unsigned char)uuid.bytes[14],
      (unsigned char)uuid.bytes[15]);
  return buf;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info_impl(int device_index) {
  // Static cache of device properties
  static auto all_devices = []() {
    // Get device count
    int count = 0;
    (void)hipGetDeviceCount(&count);

    // Collect info for all devices
    struct DeviceInfo {
      std::unordered_map<std::string, std::variant<std::string, size_t>> info;
    };

    std::vector<DeviceInfo> devices;

    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t prop;
      (void)hipGetDeviceProperties(&prop, i);

      DeviceInfo dev;
      dev.info["device_name"] = std::string(prop.name);

      // Format UUID
      dev.info["uuid"] = format_uuid(prop.uuid);

      // Architecture string (e.g., "gfx1011")
      dev.info["architecture"] = std::string(prop.gcnArchName);

      // PCI bus ID (domain:bus:device.function)
      char pci_id[32];
      snprintf(
          pci_id,
          sizeof(pci_id),
          "%04x:%02x:%02x.0",
          prop.pciDomainID,
          prop.pciBusID,
          prop.pciDeviceID);
      dev.info["pci_bus_id"] = std::string(pci_id);

      // Compute capability equivalent for AMD (GCN version)
      dev.info["compute_capability_major"] = static_cast<size_t>(prop.major);
      dev.info["compute_capability_minor"] = static_cast<size_t>(prop.minor);

      devices.push_back(std::move(dev));
    }
    return devices;
  }();

  if (device_index < 0 ||
      device_index >= static_cast<int>(all_devices.size())) {
    static auto empty =
        std::unordered_map<std::string, std::variant<std::string, size_t>>();
    return empty;
  }

  // Return a copy with fresh memory info
  // Using thread_local to avoid locks while keeping free_memory fresh
  thread_local auto device_info_copy =
      std::unordered_map<std::string, std::variant<std::string, size_t>>();

  device_info_copy = all_devices[device_index].info;

  // Get fresh memory info using hipMemGetInfo
  size_t free_mem, total_mem;

  int prev_device;
  (void)hipGetDevice(&prev_device);
  (void)hipSetDevice(device_index);
  (void)hipMemGetInfo(&free_mem, &total_mem);
  (void)hipSetDevice(prev_device);

  device_info_copy["free_memory"] = free_mem;
  device_info_copy["total_memory"] = total_mem;

  return device_info_copy;
}

} // anonymous namespace

namespace gpu {

bool is_available() {
  return true;
}

int device_count() {
  int count = 0;
  (void)hipGetDeviceCount(&count);
  return count;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  return device_info_impl(device_index);
}

} // namespace gpu

} // namespace mlx::core
