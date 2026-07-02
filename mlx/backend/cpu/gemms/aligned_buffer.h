// Copyright © 2025 Apple Inc.
#pragma once

#include <cstdint>

#include "mlx/allocator.h"

namespace mlx::core {

// Fixed-size, 32-byte aligned fp32 scratch backed by MLX's allocator.
class aligned_scratch {
 public:
  explicit aligned_scratch(size_t n_floats)
      : buf_(allocator::malloc(n_floats * sizeof(float) + alignment)) {
    auto addr = reinterpret_cast<uintptr_t>(buf_.raw_ptr());
    ptr_ = reinterpret_cast<float*>(
        (addr + (alignment - 1)) & ~static_cast<uintptr_t>(alignment - 1));
  }

  ~aligned_scratch() {
    allocator::free(buf_);
  }

  aligned_scratch(const aligned_scratch&) = delete;
  aligned_scratch& operator=(const aligned_scratch&) = delete;

  float* get() const {
    return ptr_;
  }

 private:
  static constexpr size_t alignment = 32;
  allocator::Buffer buf_;
  float* ptr_;
};

} // namespace mlx::core
