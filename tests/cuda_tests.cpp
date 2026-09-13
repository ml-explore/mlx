// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"

#ifdef _WIN32
#include "mlx/backend/cuda/wddm.h"
#endif

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

constexpr uint64_t gib = 1ULL << 30;

#ifdef _WIN32
TEST_CASE("test memory limit from budget") {
  CHECK_EQ(
      cu::memory_limit_from_budget(24 * gib, 8 * gib, 6 * gib),
      22 * gib - (24 * gib) / 20);
  CHECK_EQ(
      cu::memory_limit_from_budget(24 * gib, 4 * gib, 6 * gib),
      24 * gib - (24 * gib) / 20);
  CHECK_EQ(cu::memory_limit_from_budget(24 * gib, 25 * gib, 0), 0);
}

TEST_CASE("test wddm memory limit") {
  MemoryLimitGuard memory_limit{
      set_memory_limit(std::numeric_limits<size_t>::max())};
  cudaDeviceProp default_properties{};
  REQUIRE_EQ(cudaGetDeviceProperties(&default_properties, 0), cudaSuccess);
  auto default_limit = get_memory_limit();
  if (default_properties.memoryPoolsSupported &&
      !default_properties.integrated && !default_properties.tccDriver &&
      default_properties.luidDeviceNodeMask != 0) {
    CHECK_LT(default_limit, std::numeric_limits<size_t>::max());
  }
}
#endif // _WIN32

TEST_CASE("test clear cache trims CUDA pool") {
  if (!memory_pools_supported()) {
    return;
  }

  cudaMemPool_t pool = nullptr;
  REQUIRE_EQ(cudaDeviceGetDefaultMemPool(&pool, 0), cudaSuccess);

  CacheLimitGuard cache_limit{set_cache_limit(gib)};
  clear_cache();

  uint64_t initial_reserved = 0;
  REQUIRE_EQ(
      cudaMemPoolGetAttribute(
          pool, cudaMemPoolAttrReservedMemCurrent, &initial_reserved),
      cudaSuccess);

  auto s = default_stream(Device::gpu);
  {
    auto a = zeros({16 * 1024 * 1024}, float32, s);
    eval(a);
    synchronize(s);
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
