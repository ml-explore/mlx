// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/windows_memory.h"
#include "mlx/backend/gpu/device_info.h"

#include <dxgi1_4.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>
#include <mutex>

namespace mlx::core::cu {

size_t compute_wddm_memory_limit(
    size_t memory_limit,
    uint64_t budget,
    uint64_t usage,
    uint64_t pool_reserved) {
  auto non_pool_usage = usage > pool_reserved ? usage - pool_reserved : 0;
  auto margin = budget / 20;
  if (non_pool_usage >= budget || margin >= budget - non_pool_usage) {
    return 0;
  }

  auto wddm_limit = budget - non_pool_usage - margin;
  auto size_limit = static_cast<size_t>(
      std::min<uint64_t>(wddm_limit, std::numeric_limits<size_t>::max()));
  return std::min(memory_limit, size_limit);
}

namespace {

struct DeviceMemoryBudget {
  Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
  UINT node{0};
  std::chrono::steady_clock::time_point next_query{};
  size_t memory_limit{std::numeric_limits<size_t>::max()};
};

class WddmMemoryBudget {
 public:
  explicit WddmMemoryBudget(size_t device_count)
      : device_budgets_(device_count) {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
      return;
    }

    for (size_t device = 0; device < device_count; ++device) {
      cudaDeviceProp properties{};
      if (cudaGetDeviceProperties(&properties, static_cast<int>(device)) !=
              cudaSuccess ||
          properties.integrated || properties.tccDriver ||
          properties.luidDeviceNodeMask == 0) {
        continue;
      }

      LUID luid{};
      std::memcpy(&luid, properties.luid, sizeof(luid));
      if (luid.HighPart == 0 && luid.LowPart == 0) {
        continue;
      }

      auto& budget = device_budgets_[device];
      if (FAILED(factory->EnumAdapterByLuid(
              luid, IID_PPV_ARGS(&budget.adapter)))) {
        continue;
      }
      auto mask = properties.luidDeviceNodeMask;
      while ((mask & 1) == 0) {
        ++budget.node;
        mask >>= 1;
      }
    }
  }

  size_t
  get_memory_limit(size_t hard_memory_limit, int device, cudaMemPool_t pool) {
    if (device < 0 || device >= static_cast<int>(device_budgets_.size())) {
      return hard_memory_limit;
    }

    std::lock_guard lock(mutex_);
    auto& budget = device_budgets_[device];
    if (!budget.adapter || !pool) {
      return hard_memory_limit;
    }

    auto now = std::chrono::steady_clock::now();
    if (now < budget.next_query) {
      return std::min(hard_memory_limit, budget.memory_limit);
    }
    budget.next_query = now + std::chrono::milliseconds(20);

    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    uint64_t pool_reserved = 0;
    if (FAILED(budget.adapter->QueryVideoMemoryInfo(
            budget.node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info)) ||
        info.Budget == 0 ||
        cudaMemPoolGetAttribute(
            pool, cudaMemPoolAttrReservedMemCurrent, &pool_reserved) !=
            cudaSuccess) {
      budget.memory_limit = std::numeric_limits<size_t>::max();
      return hard_memory_limit;
    }

    budget.memory_limit = compute_wddm_memory_limit(
        std::numeric_limits<size_t>::max(),
        info.Budget,
        info.CurrentUsage,
        pool_reserved);
    return std::min(hard_memory_limit, budget.memory_limit);
  }

 private:
  std::vector<DeviceMemoryBudget> device_budgets_;
  std::mutex mutex_;
};

} // namespace

size_t get_windows_memory_limit(
    size_t hard_memory_limit,
    int device,
    cudaMemPool_t pool) {
  static WddmMemoryBudget budget(gpu::device_count());
  return budget.get_memory_limit(hard_memory_limit, device, pool);
}

} // namespace mlx::core::cu
