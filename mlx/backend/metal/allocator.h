// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <map>
#include <mutex>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core::metal {

using allocator::Buffer;

namespace {

class BufferCache {
 public:
  BufferCache(MTL::Device* device);
  ~BufferCache();
  void clear();

  MTL::Buffer* reuse_from_cache(size_t size);
  void recycle_to_cache(MTL::Buffer* buf);
  void release_cached_buffers(size_t min_bytes_to_free);
  size_t pool_size() {
    return pool_size_;
  }

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

  MTL::Device* device_;
  std::mutex cache_mutex_;

  std::multimap<size_t, BufferHolder*> buffer_pool_;
  BufferHolder* head_;
  BufferHolder* tail_;
  size_t pool_size_;
};

} // namespace

class MetalAllocator : public allocator::Allocator {
  /** Allocator for Metal GPUs. */
 public:
  virtual Buffer malloc(size_t size, bool allow_swap = false) override;
  virtual void free(Buffer buffer) override;
  size_t get_active_memory() {
    return active_memory_;
  };
  size_t get_peak_memory() {
    return peak_memory_;
  };
  size_t get_cache_memory() {
    return buffer_cache_.pool_size();
  };
  size_t set_cache_limit(size_t limit);
  size_t set_memory_limit(size_t limit, bool relaxed);

 private:
  MTL::Device* device_;
  MetalAllocator();
  friend MetalAllocator& allocator();

  // Caching allocator
  BufferCache buffer_cache_;

  // Allocation stats
  size_t block_limit_;
  size_t gc_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t max_pool_size_;
  bool relaxed_{true};
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
