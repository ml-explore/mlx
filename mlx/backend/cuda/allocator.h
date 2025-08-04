// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <mutex>
#include <set>
#include <utility>

namespace mlx::core::cu {

using allocator::Buffer;

// Stores cuda-managed unified memory.
struct CudaBuffer {
  void* data;
  size_t size;
};

class SmallSizePool {
 private:
  union Block {
    Block* next;
    CudaBuffer buf;
  };

  Block* buffer_{nullptr};
  void* data_{nullptr};
  Block* next_free_{nullptr};

 public:
  SmallSizePool();
  ~SmallSizePool();

  SmallSizePool(const SmallSizePool&) = delete;
  SmallSizePool& operator=(const SmallSizePool&) = delete;

  CudaBuffer* malloc();
  void free(CudaBuffer* buf);
  bool in_pool(CudaBuffer* buf);
};

class CudaAllocator : public allocator::Allocator {
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
  void cuda_free(CudaBuffer* buf);

  CudaAllocator();
  friend CudaAllocator& allocator();

  std::mutex mutex_;
  size_t memory_limit_;
  size_t max_pool_size_;
  BufferCache<CudaBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  SmallSizePool scalar_pool_;
};

CudaAllocator& allocator();

} // namespace mlx::core::cu
