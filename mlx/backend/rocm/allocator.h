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
// DecodeArena — deterministic bump allocator for HIP Graph capture
// ---------------------------------------------------------------------------
// During decode, the allocation pattern is fixed: same sizes in the same
// order every step. The arena allocates from a pre-sized contiguous buffer,
// guaranteeing identical pointers on each reset+replay cycle.
//
// Usage:
//   arena.begin(estimated_bytes);  // allocate backing buffer
//   // ... run decode step (allocations go through arena) ...
//   arena.reset();                 // rewind bump pointer for next step
//   // ... replay same step (same pointers) ...
//   arena.end();                   // release backing buffer

class DecodeArena {
 public:
  DecodeArena() = default;
  ~DecodeArena();

  // Allocate the backing buffer and enter arena mode.
  bool begin(size_t capacity_bytes);

  // Rewind the bump pointer. Next cycle returns same addresses.
  void reset();

  // Leave arena mode and free the backing buffer.
  void end();

  // Bump-allocate from the arena. Returns nullptr if inactive or exhausted.
  RocmBuffer* malloc(size_t size);

  // No-op free (bulk-freed on end()).
  void free(RocmBuffer* /*buf*/) {}

  bool active() const {
    return base_ != nullptr;
  }
  size_t used() const {
    return offset_;
  }
  size_t capacity() const {
    return capacity_;
  }

 private:
  void* base_{nullptr};
  size_t capacity_{0};
  size_t offset_{0};
  bool is_managed_{false};

  // Pre-allocated RocmBuffer descriptors (recycled on reset)
  std::vector<RocmBuffer> descriptors_;
  size_t desc_index_{0};
};

// ---------------------------------------------------------------------------
// RocmAllocator
// ---------------------------------------------------------------------------

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override;
  size_t size(Buffer buffer) const override;

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
  BufferCache<RocmBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  SlabAllocator slab_allocator_;

 public:
  // Arena mode for HIP Graph capture.
  // When active, malloc() returns deterministic addresses from the arena.
  DecodeArena& arena() {
    return arena_;
  }

 private:
  DecodeArena arena_;
};

RocmAllocator& allocator();

} // namespace mlx::core::rocm
