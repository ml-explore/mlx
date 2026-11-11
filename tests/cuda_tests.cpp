// Copyright © 2026 Apple Inc.

#include "doctest/doctest.h"
#include "mlx/mlx.h"

#ifdef _WIN32
#include "mlx/backend/cuda/wddm.h"
#endif

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

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

TEST_CASE("test eval after cuda error") {
  auto s = default_stream(Device::gpu);
  // A failed allocation sets a CUDA error on the calling thread.
  CHECK_THROWS(eval(zeros({1 << 20, 1 << 20}, float32, s)));
  // The error must not resurface in later evaluations.
  auto a = ones({4}, float32, s);
  CHECK_NOTHROW(eval(a));
  CHECK(array_equal(a, ones({4})).item<bool>());
}

TEST_CASE("test CUDA layer norm shared scratch") {
  constexpr int width = 1024;
  std::vector<float> data(width), expected(width);
  double mean_value = 0;
  for (int i = 0; i < width; ++i) {
    data[i] = 2.0f + (i % 31 - 15) * 0.125f;
    mean_value += data[i];
  }
  mean_value /= width;
  double variance = 0;
  for (float value : data) {
    variance += (value - mean_value) * (value - mean_value);
  }
  const double inv = 1.0 / std::sqrt(variance / width + 1e-5);
  for (int i = 0; i < width; ++i) {
    expected[i] = (data[i] - mean_value) * inv;
  }
  auto x = array(data.data(), {1, width});
  auto reference = array(expected.data(), {1, width});
  auto w = ones({width});
  auto b = zeros({width});
  eval(x, reference, w, b);
  for (int i = 0; i < 8; ++i) {
    auto y = fast::layer_norm(x, w, b, 1e-5f);
    CHECK(allclose(y, reference, 3e-5f, 3e-5f).item<bool>());
  }
}

TEST_CASE("test CUDA layer norm forward and vjp") {
  for (auto dtype : {float32, float16, bfloat16}) {
    const float tolerance = dtype == float32 ? 1e-4f
        : dtype == float16                   ? 1e-2f
                                             : 8e-2f;
    for (int width : {32, 128, 1024, 4096}) {
      CAPTURE(dtype);
      CAPTURE(width);
      auto x = astype(reshape(arange(3 * width), {3, width}) % 31, dtype) / 8;
      auto w = astype(arange(width) % 5, dtype) / 16 + 0.75f;
      auto b = astype(arange(width) % 3, dtype) / 8;
      auto g = astype(reshape(arange(3 * width), {3, width}) % 13, dtype) / 16;
      eval(x, w, b, g);
      auto fused = [](const std::vector<array>& p) {
        return std::vector<array>{fast::layer_norm(p[0], p[1], p[2], 1e-5f)};
      };
      auto reference = [](const std::vector<array>& p) {
        auto input = astype(p[0], float32);
        auto centered = input - mean(input, -1, true);
        auto normalized =
            centered * rsqrt(mean(square(centered), -1, true) + 1e-5f);
        return std::vector<array>{
            astype(normalized, p[0].dtype()) * p[1] + p[2]};
      };
      auto actual = vjp(fused, {x, w, b}, {g});
      auto expected = vjp(reference, {x, w, b}, {g});
      CHECK(allclose(actual.first[0], expected.first[0], tolerance, tolerance)
                .item<bool>());
      for (int i = 0; i < 3; ++i) {
        CAPTURE(i);
        CHECK(
            allclose(actual.second[i], expected.second[i], tolerance, tolerance)
                .item<bool>());
      }
    }
  }
}
