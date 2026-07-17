// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
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
  size_t set_cache_limit(size_t limit);
  size_t set_memory_limit(size_t limit);
  size_t get_memory_limit();
  size_t set_wired_limit(size_t limit);
  void clear_cache();

  // We don't track hazards and command buffers can run concurrently across
  // streams, so a buffer freed on the host while a kernel is still using it
  // must not be reused yet -- otherwise a later allocation lands on the same
  // memory mid-kernel (e.g. a bf16 write showing up in an int32 buffer). Mark a
  // command buffer's buffers when its encoding ends, and retire them from its
  // completion handler so the allocator knows when they're safe to reuse.
  void mark_in_flight(const std::vector<const void*>& buffers);
  void retire_in_flight(const std::vector<const void*>& buffers);

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

  // How many in-flight command buffers still reference each buffer, and the
  // buffers that were freed while in flight and are waiting to be recycled.
  std::unordered_map<MTL::Buffer*, int> in_flight_;
  std::unordered_set<MTL::Buffer*> deferred_recycle_;

  std::mutex mutex_;
};

MetalAllocator& allocator();

} // namespace mlx::core::metal
