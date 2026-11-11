// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"

#include "mlx/backend/cuda/allocator.h"

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

// Host buffers are pinned memory read over PCIe only without concurrent
// managed access (Windows, WSL, Tegra).
bool host_memory_is_pinned() {
  int managed = 0;
  REQUIRE_EQ(
      cudaDeviceGetAttribute(&managed, cudaDevAttrConcurrentManagedAccess, 0),
      cudaSuccess);
  return !managed;
}

cudaMemoryType residency(void* ptr) {
  cudaPointerAttributes attr{};
  REQUIRE_EQ(cudaPointerGetAttributes(&attr, ptr), cudaSuccess);
  return attr.type;
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

TEST_CASE("test cached host buffer is not reused for device output") {
  if (!host_memory_is_pinned()) {
    return;
  }
  clear_cache();
  auto& alloc = cu::allocator();
  cudaStream_t stream;
  REQUIRE_EQ(cudaStreamCreate(&stream), cudaSuccess);
  constexpr size_t size = 1 << 22;
  // Put a host buffer into the size-keyed cache.
  alloc.free(alloc.malloc(size));
  auto buf = alloc.malloc_async(size, 0, stream);
  CHECK_NE(
      residency(static_cast<cu::CudaBuffer*>(buf.ptr())->data),
      cudaMemoryTypeHost);
  alloc.free(buf);
  CHECK_EQ(cudaStreamSynchronize(stream), cudaSuccess);
  CHECK_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_CASE("test host data is moved to device memory") {
  if (!host_memory_is_pinned() || !memory_pools_supported()) {
    return;
  }
  auto data = [](const array& a) {
    return static_cast<cu::CudaBuffer*>(a.buffer().ptr())->data;
  };
  std::vector<float> v(1 << 20, 2.0f);
  array a(v.begin(), {1 << 20});
  CHECK_NE(residency(data(a)), cudaMemoryTypeHost);
  CHECK_EQ(sum(a).item<float>(), 2.0f * (1 << 20));

  // Arrays for a CPU default device stay on the host.
  set_default_device(Device::cpu);
  array b(v.begin(), {1 << 20});
  set_default_device(Device::gpu);
  CHECK_NE(residency(data(b)), cudaMemoryTypeDevice);
}

TEST_CASE("test eval after cuda error") {
  auto s = default_stream(Device::gpu);
  // A failed allocation sets a CUDA error on the calling thread.
  CHECK_THROWS(eval(zeros({1 << 20, 1 << 20}, float32, s)));
  // The error must not resurface in later evaluations.
  auto a = ones({4}, float32, s);
  CHECK_NOTHROW(eval(a));
  CHECK(array_equal(a, ones({4})).item<bool>());
}
