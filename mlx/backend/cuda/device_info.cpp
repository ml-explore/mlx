// Copyright © 2026 Apple Inc.

#include "mlx/backend/gpu/device_info.h"
#include "mlx/backend/cuda/cuda.h"

#include <cuda_runtime.h>
#include <dlfcn.h>

#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace mlx::core {

namespace {

// NVML dynamic loading for accurate memory reporting
// (cudaMemGetInfo only sees current process)

typedef int nvmlReturn_t;
typedef struct nvmlDevice_st* nvmlDevice_t;
struct nvmlMemory_t {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct NVMLState {
  void* handle = nullptr;
  nvmlReturn_t (*nvmlInit_v2)() = nullptr;
  nvmlReturn_t (*nvmlDeviceGetHandleByUUID)(const char*, nvmlDevice_t*) =
      nullptr;
  nvmlReturn_t (*nvmlDeviceGetMemoryInfo)(nvmlDevice_t, nvmlMemory_t*) =
      nullptr;
};

bool nvml_init(NVMLState& nvml) {
#ifdef _WIN32
  nvml.handle = dlopen("nvml.dll", RTLD_LAZY);
  if (!nvml.handle) {
    nvml.handle = dlopen(
        "C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll", RTLD_LAZY);
  }
#else
  nvml.handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
#endif
  if (!nvml.handle)
    return false;

  nvml.nvmlInit_v2 =
      (decltype(nvml.nvmlInit_v2))dlsym(nvml.handle, "nvmlInit_v2");
  nvml.nvmlDeviceGetHandleByUUID =
      (decltype(nvml.nvmlDeviceGetHandleByUUID))dlsym(
          nvml.handle, "nvmlDeviceGetHandleByUUID");
  nvml.nvmlDeviceGetMemoryInfo = (decltype(nvml.nvmlDeviceGetMemoryInfo))dlsym(
      nvml.handle, "nvmlDeviceGetMemoryInfo");

  if (!nvml.nvmlInit_v2 || !nvml.nvmlDeviceGetHandleByUUID ||
      !nvml.nvmlDeviceGetMemoryInfo) {
    return false;
  }
  return nvml.nvmlInit_v2() == 0;
}

bool nvml_get_memory(
    NVMLState& nvml,
    const char* uuid,
    size_t* free,
    size_t* total) {
  if (!nvml.handle)
    return false;
  nvmlDevice_t device;
  if (nvml.nvmlDeviceGetHandleByUUID(uuid, &device) != 0)
    return false;
  nvmlMemory_t mem;
  if (nvml.nvmlDeviceGetMemoryInfo(device, &mem) != 0)
    return false;
  *free = mem.free;
  *total = mem.total;
  return true;
}

std::string format_uuid(const cudaUUID_t& uuid) {
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
  // Static cache of device properties including UUID (needed for NVML lookup)
  static auto all_devices = []() {
    // Get device count
    int count = 0;
    cudaGetDeviceCount(&count);

    // Collect info for all devices
    struct DeviceInfo {
      std::unordered_map<std::string, std::variant<std::string, size_t>> info;
      std::string uuid;
    };

    std::vector<DeviceInfo> devices;

    for (int i = 0; i < count; ++i) {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);

      DeviceInfo dev;
      dev.info["device_name"] = std::string(prop.name);
      dev.uuid = format_uuid(prop.uuid);
      dev.info["uuid"] = dev.uuid;

      // Architecture string (e.g., "sm_89")
      char arch[16];
      snprintf(arch, sizeof(arch), "sm_%d%d", prop.major, prop.minor);
      dev.info["architecture"] = std::string(arch);

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

      // Compute capability as size_t (to match Metal's variant type)
      dev.info["compute_capability_major"] = static_cast<size_t>(prop.major);
      dev.info["compute_capability_minor"] = static_cast<size_t>(prop.minor);

      devices.push_back(std::move(dev));
    }
    return devices;
  }();

  // Initialize NVML once for fresh memory reads
  static NVMLState nvml;
  static bool nvml_initialized = nvml_init(nvml);

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

  // Get fresh memory info - try NVML first (system-wide), fallback to
  // cudaMemGetInfo (process-level)
  size_t free_mem, total_mem;

  if (nvml_initialized &&
      nvml_get_memory(
          nvml,
          all_devices[device_index].uuid.c_str(),
          &free_mem,
          &total_mem)) {
    // NVML succeeded - use system-wide memory
  } else {
    // Fallback to cudaMemGetInfo (process-scoped)
    int prev_device;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_index);
    cudaMemGetInfo(&free_mem, &total_mem);
    cudaSetDevice(prev_device);
  }

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
  cudaGetDeviceCount(&count);
  return count;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  return device_info_impl(device_index);
}

} // namespace gpu

namespace cu {

bool is_available() {
  return true;
}

} // namespace cu

} // namespace mlx::core
