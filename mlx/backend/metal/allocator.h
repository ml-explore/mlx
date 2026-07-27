// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <array>
#include <map>
#include <mutex>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

using allocator::Buffer;

class MetalAllocator : public allocator::Allocator {
  /** Allocator for Metal GPUs. */
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;
  virtual Buffer make_buffer(void* ptr, size_t size) override;
  virtual void release(Buffer buffer) override;

  size_t get_active_memory() {
    return active_memory_;
  };
  size_t get_peak_memory() {
    return peak_memory_;
  };
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  };
  size_t get_cache_memory() {
    return buffer_cache_.cache_size();
  };
  size_t get_active_buffer_count() {
    std::unique_lock lk(mutex_);
    return num_resources_;
  };
  size_t get_cache_buffer_count() {
    std::unique_lock lk(mutex_);
    return buffer_cache_.count();
  };
  std::vector<std::pair<size_t, size_t>> get_buffer_histogram();
  size_t set_cache_limit(size_t limit);
  size_t set_memory_limit(size_t limit);
  size_t get_memory_limit();
  size_t set_wired_limit(size_t limit);
  void clear_cache();

 private:
  MTL::Device* device_;

  // The size of allocations which go on the heap until it is full. This size
  // is chosen because it is the actual minimum size of a buffer allocated from
  // the heap, a heap can have at most heap.size() / 256 buffers.
  static constexpr int small_size_ = 256;
  static constexpr int heap_size_ = 1 << 20;

  MetalAllocator(Device& d);
  ~MetalAllocator();

  friend MetalAllocator& allocator();

  NS::SharedPtr<MTL::Heap> heap_;
  ResidencySet& residency_set_;

  // Caching allocator
  BufferCache<MTL::Buffer> buffer_cache_;

  // Allocation stats
  size_t block_limit_;
  size_t gc_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t max_pool_size_;
  size_t wired_limit_{0};
  size_t num_resources_{0};
  size_t resource_limit_{0};

  // Histogram of live buffer objects by power-of-two size class,
  // indexed by class exponent (class = 1 << i). A fixed array so the
  // hot-path update cannot allocate or throw -- a map node allocation
  // here could fail after newBuffer()/num_resources_++ and desync the
  // histogram. Tracks the same population as num_resources_. Guarded
  // by mutex_.
  std::array<size_t, 64> live_by_class_{};

  static int size_class(size_t size) {
    int c = 0;
    while (c < 63 && (size_t(1) << c) < size) {
      c++;
    }
    return c;
  }
  void hist_add(size_t size) noexcept {
    live_by_class_[size_class(size)]++;
  }
  void hist_sub(size_t size) noexcept {
    auto& n = live_by_class_[size_class(size)];
    if (n > 0) {
      n--;
    }
  }

  std::mutex mutex_;
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
