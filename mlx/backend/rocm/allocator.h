// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <mutex>
#include <set>
#include <utility>
#include <vector>

namespace mlx::core::rocm {

using allocator::Buffer;

struct RocmBuffer {
  void* data;
  size_t size;
  bool is_managed;
  int device;
  // Discrete-GPU only: pinned host mirror that serves CPU reads (raw_ptr)
  // WITHOUT migrating/freeing the resident VRAM copy in `data`. No default
  // initializer (keeps RocmBuffer trivial for the SizeClassPool union); set
  // explicitly in the slab path, aggregate-init'd to null in the large-alloc
  // paths. Always null on the integrated APU (device == -1).
  void* host_shadow;
  // True while host_shadow is the authoritative copy (CPU may have written
  // through raw_ptr). gpu_ptr() flushes host_shadow -> VRAM and clears it so
  // kernels see CPU writes; raw_ptr() won't re-pull from VRAM while dirty.
  bool host_dirty;
  // For stream-ordered pool buffers: the stream the buffer was allocated/used
  // on. hipFreeAsync must run on this same (actively-executing) stream so the
  // free retires in order behind the buffer's last use and the pool reclaims it.
  void* alloc_stream;
};

// ---------------------------------------------------------------------------
// SizeClassPool — fixed-size block pool with free list
// ---------------------------------------------------------------------------

class SizeClassPool {
 public:
  SizeClassPool() = default;
  ~SizeClassPool();

  SizeClassPool(const SizeClassPool&) = delete;
  SizeClassPool& operator=(const SizeClassPool&) = delete;

  void init(size_t block_size, size_t slab_page_size);
  RocmBuffer* malloc();
  void free(RocmBuffer* buf);
  bool in_pool(RocmBuffer* buf) const;
  bool grow();

  size_t block_size() const {
    return block_size_;
  }
  size_t free_count() const {
    return free_count_;
  }
  size_t total_allocated() const {
    return backing_pages_.size() * slab_page_size_;
  }
  size_t free_memory() const {
    return free_count_ * block_size_;
  }
  bool initialized() const {
    return block_size_ > 0;
  }

 private:
  union Block {
    Block* next;
    RocmBuffer buf;
  };

  size_t block_size_{0};
  size_t slab_page_size_{0};
  bool is_managed_{false};

  std::vector<void*> backing_pages_;
  std::vector<Block*> block_arrays_;
  std::vector<size_t> blocks_per_page_;

  Block* next_free_{nullptr};
  size_t free_count_{0};
  size_t total_blocks_{0};
};

// ---------------------------------------------------------------------------
// SlabAllocator — multi-tier slab allocator for sizes <= 1MB
// ---------------------------------------------------------------------------

class SlabAllocator {
 public:
  static constexpr int kNumSizeClasses = 18;
  static constexpr size_t kMaxSlabSize = 1 << 20;

  SlabAllocator();
  ~SlabAllocator() = default;

  RocmBuffer* malloc(size_t size);
  void free(RocmBuffer* buf);
  bool in_pool(RocmBuffer* buf) const;
  bool grow(size_t size);
  void warmup();

  size_t total_allocated() const;
  size_t free_memory() const;

  static int size_class_index(size_t size);
  static size_t round_to_size_class(size_t size);

 private:
  SizeClassPool pools_[kNumSizeClasses];
};

// ---------------------------------------------------------------------------
// RocmAllocator
// ---------------------------------------------------------------------------

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override { free(buffer, /*force=*/false); }
  // force=true bypasses the graph-build deferral and actually releases the
  // buffer. The deferred-free flush must force — routing through the deferring
  // path would re-defer (the graph is active for the whole session) and leak.
  void free(Buffer buffer, bool force);
  size_t size(Buffer buffer) const override;

  // CUDA-style stream-ordered allocation. When the async pool is enabled and a
  // real stream is given for a discrete device, allocates GPU-only pool memory
  // (hipMallocAsync) freed non-blocking (hipFreeAsync). Otherwise falls back to
  // the unified path (== malloc). CPU access to pool buffers is served by the
  // existing host-shadow path (device != -1) in Buffer::raw_ptr().
  Buffer malloc_async(size_t size, int device, void* stream);
  void free_async(RocmBuffer* buf, void* stream);

  // Discrete GPU: ensure buf has an up-to-date pinned host mirror for CPU reads.
  // Keeps the VRAM copy resident (does not free it or flip device to -1).
  void ensure_host_shadow(RocmBuffer& buf);

  // Discrete GPU: if buf's host shadow was written by the CPU, copy it back to
  // VRAM so kernels (gpu_ptr) see the update. No-op otherwise.
  void flush_host_shadow(RocmBuffer& buf);

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
  size_t total_memory_{0};
  size_t free_limit_{0};
  BufferCache<RocmBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  SlabAllocator slab_allocator_;

  // Per-device hipMemPool + a dedicated free stream for stream-less frees
  // (mirrors the CUDA backend). Empty entry => device has no pool support and
  // uses the blocking path.
  std::vector<void*> mem_pools_;
  std::vector<void*> free_streams_;
};

RocmAllocator& allocator();

class CommandEncoder;
// Stream-ordered allocation bound to an encoder's device/stream. Primitives
// call this for their output buffers so transient activations come from the
// device pool (fast, non-blocking free, in-eval reuse) instead of unified mem.
Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::rocm
