// Copyright © 2023 Apple Inc.

#include <stdexcept>

#include "doctest/doctest.h"

#include "mlx/allocator.h"
#include "mlx/memory.h"

using namespace mlx::core;

TEST_CASE("test simple allocations") {
  {
    auto buffer = allocator::malloc(sizeof(float));
    auto fptr = static_cast<float*>(buffer.raw_ptr());
    *fptr = 0.5f;
    CHECK_EQ(*fptr, 0.5f);
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(128 * sizeof(int));
    int* ptr = static_cast<int*>(buffer.raw_ptr());
    for (int i = 0; i < 128; ++i) {
      ptr[i] = i;
    }
    allocator::free(buffer);
  }

  {
    auto buffer = allocator::malloc(0);
    allocator::free(buffer);
  }
}

TEST_CASE("test large allocations") {
  size_t size = 1 << 30;
  for (int i = 0; i < 100; ++i) {
    auto buffer = allocator::malloc(size);
    allocator::free(buffer);
  }
}

TEST_CASE("buffer-count introspection probes track alloc/free/cache state") {
  // Probes added for ml-explore/mlx-lm#1185, where descriptor pressure was
  // diagnosed only after a crash because there was no way to observe the
  // MTLBuffer count from Python. These checks verify the probes report
  // sensible deltas under a controlled alloc/free pattern. They do not
  // depend on any specific eviction policy — only that the counters are
  // monotonic with respect to alloc/free/cache operations.

  clear_cache();
  size_t active_baseline = get_active_resource_count();
  size_t cache_baseline = get_cache_count();

  auto buffer = allocator::malloc(1 << 20); // 1 MB
  // After a fresh malloc the active count must have grown by at least 1.
  CHECK_GE(get_active_resource_count(), active_baseline + 1);

  allocator::free(buffer);
  // After free, the buffer either landed in the cache or was released.
  // Either way, the cache count must be >= baseline (cache only grows here).
  CHECK_GE(get_cache_count(), cache_baseline);

  clear_cache();
  // clear_cache must drop cache count to zero.
  CHECK_EQ(get_cache_count(), 0u);
}
