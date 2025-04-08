// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"

#include <utility>

namespace mlx::core::mxcuda {

using allocator::Buffer;

// Stores cuda-managed memory.
struct CudaBuffer {
  void* data;
  size_t size;
};

class CudaAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const {
    return active_memory_;
  };
  size_t get_peak_memory() const {
    return peak_memory_;
  };
  void reset_peak_memory() {
    peak_memory_ = 0;
  };
  size_t get_memory_limit() {
    return memory_limit_;
  }
  size_t set_memory_limit(size_t limit) {
    std::swap(memory_limit_, limit);
    return limit;
  }

 private:
  CudaAllocator();
  friend CudaAllocator& allocator();

  size_t memory_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
};

CudaAllocator& allocator();

} // namespace mlx::core::mxcuda
