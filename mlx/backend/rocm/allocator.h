// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <mutex>
#include <set>
#include <utility>

namespace mlx::core::rocm {

using allocator::Buffer;

// Stores ROCm memory buffer.
// When managed memory is available, data is allocated with hipMallocManaged.
// Otherwise, data is allocated with hipHostMalloc (pinned host memory).
struct RocmBuffer {
  void* data;
  size_t size;
  bool is_managed;  // true if allocated with hipMallocManaged
};

class SmallSizePool {
 private:
  union Block {
    Block* next;
    RocmBuffer buf;
  };

  Block* buffer_{nullptr};
  void* data_{nullptr};
  Block* next_free_{nullptr};

 public:
  SmallSizePool();
  ~SmallSizePool();

  SmallSizePool(const SmallSizePool&) = delete;
  SmallSizePool& operator=(const SmallSizePool&) = delete;

  RocmBuffer* malloc();
  void free(RocmBuffer* buf);
  bool in_pool(RocmBuffer* buf);
};

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  void rocm_free(RocmBuffer* buf);

  RocmAllocator();
  friend RocmAllocator& allocator();

  std::mutex mutex_;
  size_t memory_limit_;
  size_t max_pool_size_;
  BufferCache<RocmBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  SmallSizePool scalar_pool_;
};

RocmAllocator& allocator();

} // namespace mlx::core::rocm
