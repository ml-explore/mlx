// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/backend/cuda/windows_memory.h"
#include "mlx/mlx.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

using namespace mlx::core;

namespace {

bool memory_pools_supported() {
  int supported = 0;
  REQUIRE_EQ(
      cudaDeviceGetAttribute(&supported, cudaDevAttrMemoryPoolsSupported, 0),
      cudaSuccess);
  return supported;
}

struct CacheLimitGuard {
  ~CacheLimitGuard() {
    set_cache_limit(previous);
  }

  size_t previous;
};

struct MemoryLimitGuard {
  ~MemoryLimitGuard() {
    set_memory_limit(previous);
  }

  size_t previous;
};

} // namespace

TEST_CASE("test WDDM memory limit") {
  constexpr uint64_t gib = 1ULL << 30;

  CHECK_EQ(
      cu::compute_wddm_memory_limit(32 * gib, 24 * gib, 8 * gib, 6 * gib),
      22 * gib - (24 * gib) / 20);
  CHECK_EQ(
      cu::compute_wddm_memory_limit(16 * gib, 24 * gib, 8 * gib, 6 * gib),
      16 * gib);
  CHECK_EQ(
      cu::compute_wddm_memory_limit(32 * gib, 24 * gib, 4 * gib, 6 * gib),
      24 * gib - (24 * gib) / 20);
  CHECK_EQ(cu::compute_wddm_memory_limit(32 * gib, 24 * gib, 25 * gib, 0), 0);

  auto multi_device_limit = cu::compute_wddm_memory_limit(
      std::numeric_limits<size_t>::max(), 24 * gib, 8 * gib, 6 * gib);
  multi_device_limit = cu::compute_wddm_memory_limit(
      multi_device_limit, 16 * gib, 4 * gib, 4 * gib);
  CHECK_EQ(multi_device_limit, 16 * gib - (16 * gib) / 20);
}

TEST_CASE("test Windows memory limit query") {
  cudaDeviceProp properties{};
  REQUIRE_EQ(cudaGetDeviceProperties(&properties, 0), cudaSuccess);

  MemoryLimitGuard memory_limit{
      set_memory_limit(std::numeric_limits<size_t>::max())};
  auto limit = get_memory_limit();
  if (properties.memoryPoolsSupported && !properties.integrated &&
      !properties.tccDriver && properties.luidDeviceNodeMask != 0) {
    CHECK_LT(limit, std::numeric_limits<size_t>::max());
  }
}

TEST_CASE("test clear cache trims CUDA pool") {
  if (!memory_pools_supported()) {
    return;
  }

  cudaMemPool_t pool = nullptr;
  REQUIRE_EQ(cudaDeviceGetDefaultMemPool(&pool, 0), cudaSuccess);

  CacheLimitGuard cache_limit{set_cache_limit(1ULL << 30)};
  clear_cache();

  uint64_t initial_reserved = 0;
  REQUIRE_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &initial_reserved),
      cudaSuccess);

  {
    auto a = zeros({16 * 1024 * 1024}, float32, Device::gpu);
    eval(a);
    synchronize();
  }
  CHECK_GE(get_cache_memory(), 64ULL << 20);

  uint64_t allocated_reserved = 0;
  REQUIRE_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &allocated_reserved),
      cudaSuccess);
  CHECK_GT(allocated_reserved, initial_reserved);

  clear_cache();
  uint64_t final_reserved = 0;
  REQUIRE_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &final_reserved),
      cudaSuccess);
  CHECK_LT(final_reserved, allocated_reserved);
}
