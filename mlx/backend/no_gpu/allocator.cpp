// Copyright © 2023-2026 Apple Inc.

#include <algorithm>
#include <mutex>

#include "mlx/allocator.h"
#include "mlx/backend/common/buffer_cache.h"
#include "mlx/memory.h"
#include "mlx/stream.h"

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

namespace {

void synchronize_cpu_streams() {
  for (const auto& s : get_streams()) {
    if (s.device == Device::cpu) {
      synchronize(s);
    }
  }
}

} // namespace

namespace allocator {

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

  size_t get_cache_memory() const;
  size_t set_cache_limit(size_t limit);
  void clear_cache();

 private:
  friend CommonAllocator& common_allocator();
  CommonAllocator();

  static size_t get_buffer_size(void* buf);

  size_t memory_limit_;
  size_t cache_limit_;
  size_t active_memory_{0};
  size_t peak_memory_{0};
  mutable std::mutex mutex_;
  mutable BufferCache<void> buffer_cache_;
};

CommonAllocator::CommonAllocator()
    : memory_limit_(0.8 * get_memory_size()),
      cache_limit_(32UL << 20), // 32 MB default cache limit
      buffer_cache_(/* page_size */ 4096, get_buffer_size, std::free) {
  if (memory_limit_ == 0) {
    memory_limit_ = 1ULL << 33;
  }
}

Buffer CommonAllocator::malloc(size_t size) {
  std::unique_lock lk(mutex_);
  // Try cache first
  void* cached = buffer_cache_.reuse_from_cache(size);
  if (cached) {
    active_memory_ += get_buffer_size(cached);
    peak_memory_ = std::max(active_memory_, peak_memory_);
    return Buffer{cached};
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

  if (sz > 0 && buffer_cache_.cache_size() + sz <= cache_limit_) {
    buffer_cache_.recycle_to_cache(buffer.ptr());
  } else {
    lk.unlock();
    std::free(buffer.ptr());
  }
}

size_t CommonAllocator::size(Buffer buffer) const {
  return get_buffer_size(buffer.ptr());
}

size_t CommonAllocator::get_cache_memory() const {
  std::unique_lock lk(mutex_);
  return buffer_cache_.cache_size();
}

size_t CommonAllocator::set_cache_limit(size_t limit) {
  synchronize_cpu_streams();
  std::unique_lock lk(mutex_);
  std::swap(cache_limit_, limit);
  if (buffer_cache_.cache_size() > cache_limit_) {
    buffer_cache_.release_cached_buffers(
        buffer_cache_.cache_size() - cache_limit_);
  }
  return limit;
}

void CommonAllocator::clear_cache() {
  {
    std::unique_lock lk(mutex_);
    if (buffer_cache_.cache_size() == 0) {
      return;
    }
  }
  synchronize_cpu_streams();
  std::unique_lock lk(mutex_);
  buffer_cache_.clear();
}

size_t CommonAllocator::get_buffer_size(void* buf) {
  if (buf) {
    return *static_cast<size_t*>(buf);
  } else {
    return 0;
  }
}

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
