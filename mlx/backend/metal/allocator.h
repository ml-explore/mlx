// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/resident.h"

namespace mlx::core::metal {

using allocator::Buffer;

namespace {

class BufferCache {
 public:
  BufferCache(ResidencySet& residency_set);
  ~BufferCache();

  MTL::Buffer* reuse_from_cache(size_t size);
  void recycle_to_cache(MTL::Buffer* buf);
  int release_cached_buffers(size_t min_bytes_to_free);
  size_t cache_size() {
    return pool_size_;
  }
  int clear();

 private:
  struct BufferHolder {
   public:
    BufferHolder(MTL::Buffer* buf_) : buf(buf_), prev(nullptr), next(nullptr) {}

    BufferHolder* prev;
    BufferHolder* next;
    MTL::Buffer* buf;
  };

  void add_at_head(BufferHolder* to_add);
  void remove_from_list(BufferHolder* to_remove);

  std::multimap<size_t, BufferHolder*> buffer_pool_;
  BufferHolder* head_;
  BufferHolder* tail_;
  size_t pool_size_;
  ResidencySet& residency_set_;
};

} // namespace

class MetalAllocator : public allocator::Allocator {
  /** Allocator for Metal GPUs. */
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;
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
  MTL::Heap* heap_;
  MetalAllocator();
  ~MetalAllocator();
  friend MetalAllocator& allocator();

  // Caching allocator
  BufferCache buffer_cache_;

  ResidencySet residency_set_;

  // Allocation stats
  size_t block_limit_;
  size_t gc_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t max_pool_size_;
  size_t wired_limit_{0};
  bool relaxed_{true};
  size_t num_resources_{0};
  size_t resource_limit_{0};

  std::mutex mutex_;
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
