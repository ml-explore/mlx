// Copyright © 2023 Apple Inc.

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

 private:
  MTL::Device* device_;
  MetalAllocator();
  friend MetalAllocator& allocator();

  // Caching allocator
  BufferCache buffer_cache_;

  // Allocation stats
  size_t peak_allocated_size_;
  size_t block_limit_;
  size_t gc_limit_;
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
