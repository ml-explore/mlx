// Copyright © 2023-2026 Apple Inc.

#include <algorithm>
#include <mutex>

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/memory.h"

#ifdef __APPLE__
#include "mlx/backend/no_gpu/apple_memory.h"
#elif defined(__linux__)
#include "mlx/backend/no_gpu/linux_memory.h"
#else
size_t get_memory_size() {
  return 0;
}
#endif

namespace mlx::core {

namespace allocator {

struct CpuCachedBuffer {
  void* ptr;
  size_t size;
  CpuCachedBuffer* next_free; // intrusive freelist for object pooling
};

class CommonAllocator : public Allocator {
  /** A general CPU allocator with buffer caching. */
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;

  size_t get_active_memory() const {
    return active_memory_;
  };
  size_t get_peak_memory() const {
    return peak_memory_;
  };
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  };
  size_t get_memory_limit() {
    return memory_limit_;
  }
  size_t set_memory_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(memory_limit_, limit);
    return limit;
  }

  size_t get_cache_memory() const {
    return buffer_cache_.cache_size();
  }
  size_t set_cache_limit(size_t limit) {
    std::unique_lock lk(mutex_);
    std::swap(cache_limit_, limit);
    if (buffer_cache_.cache_size() > cache_limit_) {
      buffer_cache_.release_cached_buffers(
          buffer_cache_.cache_size() - cache_limit_);
    }
    return limit;
  }
  void clear_cache() {
    std::unique_lock lk(mutex_);
    buffer_cache_.clear();
  }

 private:
  CpuCachedBuffer* pool_head_ = nullptr;

  CpuCachedBuffer* alloc_ccb(void* ptr, size_t sz) {
    CpuCachedBuffer* ccb;
    if (pool_head_) {
      ccb = pool_head_;
      pool_head_ = ccb->next_free;
      ccb->ptr = ptr;
      ccb->size = sz;
    } else {
      ccb = new CpuCachedBuffer{ptr, sz, nullptr};
    }
    return ccb;
  }

  void free_ccb(CpuCachedBuffer* ccb) {
    ccb->next_free = pool_head_;
    pool_head_ = ccb;
  }

  size_t memory_limit_;
  size_t cache_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  mutable std::mutex mutex_;
  mutable BufferCache<CpuCachedBuffer> buffer_cache_;
  CommonAllocator()
      : memory_limit_(0.8 * get_memory_size()),
        cache_limit_(32UL << 20), // 32 MB default cache limit
        buffer_cache_(
            /* page_size = */
            4096,
            /* get_size = */
            [](CpuCachedBuffer* b) { return b->size; },
            /* free = */
            [this](CpuCachedBuffer* b) {
              std::free(b->ptr);
              free_ccb(b);
            }) {
    if (memory_limit_ == 0) {
      memory_limit_ = 1ULL << 33;
    }
  };

  friend CommonAllocator& common_allocator();
};

CommonAllocator& common_allocator() {
  static CommonAllocator allocator_;
  return allocator_;
}

Allocator& allocator() {
  return common_allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<size_t*>(ptr_) + 1;
}

Buffer CommonAllocator::malloc(size_t size) {
  std::unique_lock lk(mutex_);
  // Try cache first
  CpuCachedBuffer* cached = buffer_cache_.reuse_from_cache(size);
  if (cached) {
    void* ptr = cached->ptr;
    free_ccb(cached);
    // Update size header to reflect requested size
    *static_cast<size_t*>(ptr) = size;
    active_memory_ += size;
    peak_memory_ = std::max(active_memory_, peak_memory_);
    return Buffer{ptr};
  }
  lk.unlock();

  // Cache miss: allocate from OS
  void* ptr = std::malloc(size + sizeof(size_t));
  if (ptr != nullptr) {
    *static_cast<size_t*>(ptr) = size;
  }
  lk.lock();
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{ptr};
}

void CommonAllocator::free(Buffer buffer) {
  auto sz = size(buffer);
  std::unique_lock lk(mutex_);
  active_memory_ -= sz;

  if (buffer_cache_.cache_size() + sz <= cache_limit_) {
    // Add to cache for reuse
    auto* entry = alloc_ccb(buffer.ptr(), sz);
    buffer_cache_.recycle_to_cache(entry);
  } else {
    lk.unlock();
    std::free(buffer.ptr());
  }
}

size_t CommonAllocator::size(Buffer buffer) const {
  if (buffer.ptr() == nullptr) {
    return 0;
  }
  return *static_cast<size_t*>(buffer.ptr());
}

} // namespace allocator

size_t get_active_memory() {
  return allocator::common_allocator().get_active_memory();
}
size_t get_peak_memory() {
  return allocator::common_allocator().get_peak_memory();
}
void reset_peak_memory() {
  return allocator::common_allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return allocator::common_allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return allocator::common_allocator().get_memory_limit();
}

size_t get_cache_memory() {
  return allocator::common_allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return allocator::common_allocator().set_cache_limit(limit);
}
size_t set_wired_limit(size_t) {
  return 0;
}
void clear_cache() {
  allocator::common_allocator().clear_cache();
}

} // namespace mlx::core
