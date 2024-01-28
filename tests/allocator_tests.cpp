// Copyright © 2023 Apple Inc.

#include <stdexcept>

#include "doctest/doctest.h"

#include "mlx/allocator.h"

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
