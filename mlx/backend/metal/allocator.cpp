// Copyright © 2023 Apple Inc.

#include "mlx/backend/metal/allocator.h"
#include "mlx/backend/metal/metal.h"

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
  clear();
}

void BufferCache::clear() {
  std::lock_guard<std::mutex> lk(cache_mutex_);
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
  std::lock_guard<std::mutex> lk(cache_mutex_);

  // Find the closest buffer in pool
  MTL::Buffer* pbuf = nullptr;

  // Make sure we use > 50% of the available memory
  if (auto it = buffer_pool_.lower_bound(size);
      it != buffer_pool_.end() && it->first < 2 * size) {
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
  std::lock_guard<std::mutex> lk(cache_mutex_);

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
    std::lock_guard<std::mutex> lk(cache_mutex_);
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
  if (!to_remove)
    return;

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
      buffer_cache_(device_),
      peak_allocated_size_(0),
      block_limit_(1.5 * device_->recommendedMaxWorkingSetSize()) {}

Buffer MetalAllocator::malloc(size_t size) {
  // Align up memory
  if (size > vm_page_size) {
    size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);
  }

  MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);

  // Prepare to allocate new memory as needed
  if (!buf) {
    // First check if the cache is big but nothing fits, garbage collect
    // if so
    // TODO maybe block limit and gc limit should be different
    if (buffer_cache_.size() >= block_limit_) {
      buffer_cache_.release_cached_buffers(
          std::max(buffer_cache_.size() - block_limit_, size));
    }

    // If there is still too much memory pressure, fail (likely causes a wait).
    if (device_->currentAllocatedSize() >= block_limit_) {
      return Buffer{nullptr};
    }

    // Allocate new buffer if needed
    size_t res_opt = MTL::ResourceStorageModeShared;
    res_opt |= MTL::ResourceHazardTrackingModeTracked;
    buf = device_->newBuffer(size, res_opt);
  }

  peak_allocated_size_ =
      std::max(peak_allocated_size_, device_->currentAllocatedSize());

  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::free(Buffer buffer) {
  auto buf = static_cast<MTL::Buffer*>(buffer.ptr());
  buffer_cache_.recycle_to_cache(buf);
}

MetalAllocator& allocator() {
  static MetalAllocator allocator_;
  return allocator_;
}

} // namespace metal

} // namespace mlx::core
