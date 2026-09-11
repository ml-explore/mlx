// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/wddm.h"
#include "mlx/backend/cuda/utils.h"
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

namespace {

struct DeviceMemoryBudget {
  Microsoft::WRL::ComPtr<IDXGIAdapter3> adapter;
  UINT node{0};
  std::chrono::steady_clock::time_point next_query;
  size_t memory_limit{std::numeric_limits<size_t>::max()};
};

class WddmMemoryBudget {
 public:
  explicit WddmMemoryBudget(size_t device_count) : budgets_(device_count) {
    Microsoft::WRL::ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) {
      return;
    }

    for (int device = 0; device < device_count; ++device) {
      cudaDeviceProp properties{};
      CHECK_CUDA_ERROR(cudaGetDeviceProperties(&properties, device));
      if (properties.integrated || properties.tccDriver ||
          properties.luidDeviceNodeMask == 0) {
        continue;
      }

      LUID luid{};
      std::memcpy(&luid, properties.luid, sizeof(luid));
      if (luid.HighPart == 0 && luid.LowPart == 0) {
        continue;
      }

      auto& budget = budgets_[device];
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

  size_t get_memory_limit(int device, cudaMemPool_t pool) {
    size_t max_size = std::numeric_limits<size_t>::max();
    auto& budget = budgets_.at(device);
    if (!budget.adapter || !pool) {
      return max_size;
    }

    auto now = std::chrono::steady_clock::now();
    std::lock_guard lock(mutex_);
    if (now < budget.next_query) {
      return budget.memory_limit;
    }
    budget.next_query = now + std::chrono::milliseconds(20);

    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    uint64_t pool_reserved = 0;
    if (FAILED(budget.adapter->QueryVideoMemoryInfo(
            budget.node, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
      return max_size;
    }
    if (info.Budget == 0) {
      return max_size;
    }

    CHECK_CUDA_ERROR(cudaMemPoolGetAttribute(
        pool, cudaMemPoolAttrReservedMemCurrent, &pool_reserved));
    budget.memory_limit =
        memory_limit_from_budget(info.Budget, info.CurrentUsage, pool_reserved);
    return budget.memory_limit;
  }

 private:
  std::vector<DeviceMemoryBudget> budgets_;
  std::mutex mutex_;
};

} // namespace

size_t get_wddm_memory_limit(int device, cudaMemPool_t pool) {
  static WddmMemoryBudget budget(gpu::device_count());
  return budget.get_memory_limit(device, pool);
}

} // namespace mlx::core::cu
