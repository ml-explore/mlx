// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"

#include <deque>
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
  // WITHOUT migrating/freeing the resident VRAM copy in `data`.
  void* host_shadow;
  bool host_dirty;
  // Stream the buffer was allocated/used on (for hipFreeAsync ordering).
  void* alloc_stream;
};

// ---------------------------------------------------------------------------
// DecodeArena — deterministic bump allocator for build-once HIP-graph decode.
// ---------------------------------------------------------------------------
struct DecodeArena {
  void* base{nullptr};
  size_t capacity{0};
  size_t offset{0};
  bool active{false};
  bool overflowed{false};
  int device{-1};
  std::deque<RocmBuffer> wrappers;
  size_t next_wrapper{0};
  size_t high_water{0};
  size_t floor_offset{0};
  size_t floor_wrapper{0};

  bool contains(const void* p) const {
    return base && p >= base &&
        p < static_cast<const char*>(base) + capacity;
  }
};

// ---------------------------------------------------------------------------
// SizeClassPool / SlabAllocator — small fixed-size blocks (ROCm-only helper;
// CUDA uses a tiny scalar pool instead). Kept for ≤1MB fast path.
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
// RocmAllocator — CUDA-aligned memory management
//
// Matches mlx/backend/cuda/allocator.cpp:
//   malloc_async: BufferCache hit → else hipMallocAsync / hipMalloc
//   free:         recycle BufferCache → else hipFreeAsync / hipFree
//   max_pool_size_ defaults to memory_limit_ (host free-list can hold reused
//   buffers up to the limit; real free only under pressure / over cache cap).
// ---------------------------------------------------------------------------

class RocmAllocator : public allocator::Allocator {
 public:
  Buffer malloc(size_t size) override;
  void free(Buffer buffer) override {
    free(buffer, /*force=*/false);
  }
  void free(Buffer buffer, bool force);
  size_t size(Buffer buffer) const override;

  Buffer make_buffer(void* ptr, size_t size) override;
  void release(Buffer buffer) override;

  // CUDA: Buffer malloc_async(size, device, stream)
  Buffer malloc_async(size_t size, int device, void* stream);

  // CUDA-style: free device memory only (does not delete RocmBuffer shell).
  void free_device(RocmBuffer& buf, void* stream = nullptr);
  // CUDA free_cuda_buffer: free device mem + delete shell (BufferCache free_).
  void free_rocm_buffer(RocmBuffer* buf);

  void ensure_host_shadow(RocmBuffer& buf);
  void flush_host_shadow(RocmBuffer& buf);

  bool decode_arena_begin(size_t capacity, int device, void* stream);
  void decode_arena_reset();
  void decode_arena_freeze_floor();
  void decode_arena_reset_to_floor();
  void decode_arena_end();
  bool decode_arena_active() const {
    return decode_arena_.active;
  }
  size_t decode_arena_high_water() const {
    return decode_arena_.high_water;
  }
  bool decode_arena_overflowed() const {
    return decode_arena_.overflowed;
  }

  bool train_arena_begin(size_t capacity, int device, void* stream) {
    return decode_arena_begin(capacity, device, stream);
  }
  void train_arena_reset() {
    decode_arena_reset();
  }
  void train_arena_freeze_floor() {
    decode_arena_freeze_floor();
  }
  void train_arena_reset_to_floor() {
    decode_arena_reset_to_floor();
  }
  void train_arena_end() {
    decode_arena_end();
  }
  bool train_arena_active() const {
    return decode_arena_active();
  }
  size_t train_arena_high_water() const {
    return decode_arena_high_water();
  }
  bool train_arena_overflowed() const {
    return decode_arena_overflowed();
  }

  size_t get_active_memory() const;
  size_t get_peak_memory() const;
  void reset_peak_memory();
  size_t get_memory_limit();
  size_t set_memory_limit(size_t limit);
  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  RocmAllocator();
  friend RocmAllocator& allocator();

  std::mutex mutex_;
  size_t memory_limit_;
  size_t free_limit_;
  size_t total_memory_{0};
  size_t max_pool_size_;
  BufferCache<RocmBuffer> buffer_cache_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  SlabAllocator slab_allocator_;
  DecodeArena decode_arena_;
  RocmBuffer* arena_alloc(size_t size);

  std::vector<void*> mem_pools_;
  std::vector<void*> free_streams_;
};

RocmAllocator& allocator();

class CommandEncoder;
Buffer malloc_async(size_t size, CommandEncoder& encoder);

} // namespace mlx::core::rocm
