// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/metal/metal_impl.h"

#include <mach/vm_page_size.h>
#include <unistd.h>
#include <cstdlib>

namespace mlx::core {

namespace allocator {

Allocator& allocator() {
  return metal::allocator();
}

void* Buffer::raw_ptr() {
  return static_cast<MTL::Buffer*>(ptr_)->contents();
}

} // namespace allocator

namespace metal {

namespace {

BufferCache::BufferCache(MTL::Device* device)
    : device_(device), head_(nullptr), tail_(nullptr), pool_size_(0) {}

BufferCache::~BufferCache() {
  auto thread_pool = metal::new_scoped_memory_pool();
  clear();
}

void BufferCache::clear() {
  for (auto& [size, holder] : buffer_pool_) {
    if (holder->buf)
      holder->buf->release();
    delete holder;
  }
  buffer_pool_.clear();
  pool_size_ = 0;
  head_ = nullptr;
  tail_ = nullptr;
}

MTL::Buffer* BufferCache::reuse_from_cache(size_t size) {
  // Find the closest buffer in pool
  MTL::Buffer* pbuf = nullptr;

  auto it = buffer_pool_.lower_bound(size);

  // Make sure we use most of the available memory
  while (!pbuf && it != buffer_pool_.end() &&
         it->first < std::min(2 * size, size + 2 * vm_page_size)) {
    // Collect from the cache
    pbuf = it->second->buf;

    // Remove from cache
    remove_from_list(it->second);
    delete it->second;
    it = buffer_pool_.erase(it);
  }

  if (pbuf) {
    pool_size_ -= pbuf->length();
  }

  return pbuf;
}

void BufferCache::recycle_to_cache(MTL::Buffer* buf) {
  // Add to cache
  if (buf) {
    BufferHolder* bh = new BufferHolder(buf);
    add_at_head(bh);
    pool_size_ += buf->length();
    buffer_pool_.insert({buf->length(), bh});
  }
}

void BufferCache::release_cached_buffers(size_t min_bytes_to_free) {
  if (min_bytes_to_free >= 0.9 * pool_size_) {
    clear();
  } else {
    size_t total_bytes_freed = 0;

    while (tail_ && (total_bytes_freed < min_bytes_to_free)) {
      if (tail_->buf) {
        total_bytes_freed += tail_->buf->length();
        tail_->buf->release();
        tail_->buf = nullptr;
      }
      remove_from_list(tail_);
    }
    pool_size_ -= total_bytes_freed;
  }
}

void BufferCache::add_at_head(BufferCache::BufferHolder* to_add) {
  if (!to_add)
    return;

  if (!head_) {
    head_ = to_add;
    tail_ = to_add;
  } else {
    head_->prev = to_add;
    to_add->next = head_;
    head_ = to_add;
  }
}

void BufferCache::remove_from_list(BufferCache::BufferHolder* to_remove) {
  if (!to_remove) {
    return;
  }

  // If in the middle
  if (to_remove->prev && to_remove->next) {
    to_remove->prev->next = to_remove->next;
    to_remove->next->prev = to_remove->prev;
  } else if (to_remove->prev && to_remove == tail_) { // If tail
    tail_ = to_remove->prev;
    tail_->next = nullptr;
  } else if (to_remove == head_ && to_remove->next) { // If head
    head_ = to_remove->next;
    head_->prev = nullptr;
  } else if (to_remove == head_ && to_remove == tail_) { // If only element
    head_ = nullptr;
    tail_ = nullptr;
  }

  to_remove->prev = nullptr;
  to_remove->next = nullptr;
}

} // namespace

MetalAllocator::MetalAllocator()
    : device_(device(mlx::core::Device::gpu).mtl_device()),
      buffer_cache_(device_) {
  auto memsize = std::get<size_t>(device_info()["memory_size"]);
  block_limit_ =
      std::min(1.5 * device_->recommendedMaxWorkingSetSize(), 0.95 * memsize);
  gc_limit_ = std::min(
      static_cast<size_t>(0.95 * device_->recommendedMaxWorkingSetSize()),
      block_limit_);
  max_pool_size_ = block_limit_;
}

size_t MetalAllocator::set_cache_limit(size_t limit) {
  std::swap(limit, max_pool_size_);
  return limit;
};

size_t MetalAllocator::set_memory_limit(size_t limit, bool relaxed) {
  std::swap(limit, block_limit_);
  relaxed_ = relaxed;
  gc_limit_ = std::min(
      block_limit_,
      static_cast<size_t>(0.95 * device_->recommendedMaxWorkingSetSize()));
  return limit;
};

Buffer MetalAllocator::malloc(size_t size, bool allow_swap /* = false */) {
  // Metal doesn't like empty buffers
  if (size == 0) {
    return Buffer{nullptr};
  }

  // More helpful message if maximum buffer length is exceeded
  if (size > device_->maxBufferLength()) {
    std::ostringstream msg;
    msg << "Attempting to allocate " << size << " bytes which is greater than"
        << " the maximum allowed buffer size of " << device_->maxBufferLength()
        << " bytes.";
    throw std::runtime_error(msg.str());
  }

  // Align up memory
  if (size > vm_page_size) {
    size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);
  }

  // Try the cache
  std::unique_lock lk(mutex_);
  MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    size_t mem_required = get_active_memory() + get_cache_memory() + size;

    // If there is too much memory pressure, fail (likely causes a wait).
    if (!(allow_swap && relaxed_) && mem_required >= block_limit_) {
      return Buffer{nullptr};
    }

    auto thread_pool = metal::new_scoped_memory_pool();

    // If we have a lot of memory pressure or are over the maximum cache size,
    // try to reclaim memory from the cache
    if (mem_required >= gc_limit_) {
      buffer_cache_.release_cached_buffers(mem_required - gc_limit_);
    }

    // Allocate new buffer if needed
    size_t res_opt = MTL::ResourceStorageModeShared;
    res_opt |= MTL::ResourceHazardTrackingModeTracked;
    lk.unlock();
    buf = device_->newBuffer(size, res_opt);
    lk.lock();
  }

  active_memory_ += buf->length();
  peak_memory_ = std::max(peak_memory_, active_memory_);

  // Maintain the cache below the requested limit
  if (get_cache_memory() >= max_pool_size_) {
    auto thread_pool = metal::new_scoped_memory_pool();
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }

  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::clear_cache() {
  std::unique_lock lk(mutex_);
  buffer_cache_.clear();
}

void MetalAllocator::free(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  std::unique_lock lk(mutex_);
  active_memory_ -= buf->length();
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    lk.unlock();
    auto thread_pool = metal::new_scoped_memory_pool();
    buf->release();
  }
}

size_t MetalAllocator::size(Buffer buffer) const {
  return static_cast<MTL::Buffer*>(buffer.ptr())->length();
}

MetalAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of MetalAllocator will
  // not be called on exit and all the buffers will be leaked. This is necessary
  // because releasing buffers can take more than 30sec when the program holds a
  // lot of RAM (for example inferencing a LLM), and it would feel frozen to
  // users when exiting.
  // TODO(zcbenz): Consider using the `base::NoDestructor` class from Chromium
  // when applying this pattern to more places, or when introducing sanitizers
  // to MLX.
  // https://source.chromium.org/chromium/chromium/src/+/main:base/no_destructor.h
  static MetalAllocator* allocator_ = new MetalAllocator;
  return *allocator_;
}

size_t set_cache_limit(size_t limit) {
  return allocator().set_cache_limit(limit);
}
size_t set_memory_limit(size_t limit, bool relaxed /* = true */) {
  return allocator().set_memory_limit(limit, relaxed);
}
size_t get_active_memory() {
  return allocator().get_active_memory();
}
size_t get_peak_memory() {
  return allocator().get_peak_memory();
}
void reset_peak_memory() {
  allocator().reset_peak_memory();
}
size_t get_cache_memory() {
  return allocator().get_cache_memory();
}
void clear_cache() {
  return allocator().clear_cache();
}

} // namespace metal

} // namespace mlx::core
