// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cuda.h"

#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace mlx::core::cu {

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
  bool initialized = false;
};

NVMLState& nvml_state() {
  static NVMLState state;
  return state;
}

std::mutex& nvml_mutex() {
  static std::mutex m;
  return m;
}

bool nvml_init() {
  std::lock_guard<std::mutex> lock(nvml_mutex());
  auto& nvml = nvml_state();
  if (nvml.initialized)
    return nvml.handle != nullptr;
  nvml.initialized = true;

#ifdef _WIN32
  nvml.handle = LoadLibraryA("nvml.dll");
  if (!nvml.handle) {
    nvml.handle =
        LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
  }
  if (!nvml.handle)
    return false;
  nvml.nvmlInit_v2 = (decltype(nvml.nvmlInit_v2))GetProcAddress(
      (HMODULE)nvml.handle, "nvmlInit_v2");
  nvml.nvmlDeviceGetHandleByUUID =
      (decltype(nvml.nvmlDeviceGetHandleByUUID))GetProcAddress(
          (HMODULE)nvml.handle, "nvmlDeviceGetHandleByUUID");
  nvml.nvmlDeviceGetMemoryInfo =
      (decltype(nvml.nvmlDeviceGetMemoryInfo))GetProcAddress(
          (HMODULE)nvml.handle, "nvmlDeviceGetMemoryInfo");
#else
  nvml.handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
  if (!nvml.handle)
    return false;
  nvml.nvmlInit_v2 =
      (decltype(nvml.nvmlInit_v2))dlsym(nvml.handle, "nvmlInit_v2");
  nvml.nvmlDeviceGetHandleByUUID =
      (decltype(nvml.nvmlDeviceGetHandleByUUID))dlsym(
          nvml.handle, "nvmlDeviceGetHandleByUUID");
  nvml.nvmlDeviceGetMemoryInfo = (decltype(nvml.nvmlDeviceGetMemoryInfo))dlsym(
      nvml.handle, "nvmlDeviceGetMemoryInfo");
#endif

  if (!nvml.nvmlInit_v2 || !nvml.nvmlDeviceGetHandleByUUID ||
      !nvml.nvmlDeviceGetMemoryInfo) {
    return false;
  }
  return nvml.nvmlInit_v2() == 0;
}

bool nvml_get_memory(const char* uuid, size_t* free, size_t* total) {
  auto& nvml = nvml_state();
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

} // anonymous namespace

bool is_available() {
  return true;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  // Cache per device (static info doesn't change, memory refreshed)
  static std::unordered_map<
      int,
      std::unordered_map<std::string, std::variant<std::string, size_t>>>
      cache;
  static std::mutex cache_mutex;

  std::lock_guard<std::mutex> lock(cache_mutex);

  auto it = cache.find(device_index);
  if (it != cache.end()) {
    // Refresh memory info
    std::string uuid = std::get<std::string>(it->second["uuid"]);
    size_t free_mem, total_mem;
    if (nvml_get_memory(uuid.c_str(), &free_mem, &total_mem)) {
      it->second["free_memory"] = free_mem;
      it->second["total_memory"] = total_mem;
    }
    return it->second;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_index);

  auto& info = cache[device_index];
  info["device_name"] = std::string(prop.name);
  info["uuid"] = format_uuid(prop.uuid);

  // Architecture string (e.g., "sm_89")
  char arch[16];
  snprintf(arch, sizeof(arch), "sm_%d%d", prop.major, prop.minor);
  info["architecture"] = std::string(arch);

  // PCI bus ID (domain:bus:device.function)
  char pci_id[32];
  snprintf(
      pci_id,
      sizeof(pci_id),
      "%04x:%02x:%02x.0",
      prop.pciDomainID,
      prop.pciBusID,
      prop.pciDeviceID);
  info["pci_bus_id"] = std::string(pci_id);

  // Compute capability as size_t (to match Metal's variant type)
  info["compute_capability_major"] = static_cast<size_t>(prop.major);
  info["compute_capability_minor"] = static_cast<size_t>(prop.minor);

  // Memory - try NVML first, fallback to cudaMemGetInfo
  nvml_init();
  size_t free_mem, total_mem;
  std::string uuid = std::get<std::string>(info["uuid"]);
  if (!nvml_get_memory(uuid.c_str(), &free_mem, &total_mem)) {
    int prev_device;
    cudaGetDevice(&prev_device);
    cudaSetDevice(device_index);
    cudaMemGetInfo(&free_mem, &total_mem);
    cudaSetDevice(prev_device);
  }
  info["free_memory"] = free_mem;
  info["total_memory"] = total_mem;

  return cache[device_index];
}

} // namespace mlx::core::cu
