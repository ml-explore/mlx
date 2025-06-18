// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/worker.h"

#include <fmt/format.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

RocmAllocator::RocmAllocator()
    : buffer_cache_(
          getpagesize(),
          [](RocmBuffer* buf) { return buf->size; },
          [this](RocmBuffer* buf) {
            rocm_free(buf->data);
            delete buf;
          }) {
  // TODO: Set memory limit for multi-device.
  size_t free, total;
  CHECK_HIP_ERROR(hipMemGetInfo(&free, &total));
  memory_limit_ = total * 0.8;
  max_pool_size_ = memory_limit_;
}

Buffer RocmAllocator::malloc(size_t size) {
  // Find available buffer from cache.
  std::unique_lock lock(mutex_);
  RocmBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    // If we have a lot of memory pressure or are over the maximum cache size,
    // try to reclaim memory from the cache.
    size_t mem_required = get_active_memory() + get_cache_memory() + size;
    if (mem_required >= memory_limit_) {
      buffer_cache_.release_cached_buffers(mem_required - memory_limit_);
    }

    lock.unlock();
    buf = new RocmBuffer{nullptr, size};
    hipError_t err = hipMallocManaged(&buf->data, size);
    if (err != hipSuccess && err != hipErrorMemoryAllocation) {
      throw std::runtime_error(
          fmt::format("hipMallocManaged failed: {}.", hipGetErrorString(err)));
    }
    lock.lock();
  }
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Maintain the cache below the requested limit.
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }

  return Buffer{buf};
}

void RocmAllocator::free(Buffer buffer) {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    lock.unlock();
    rocm_free(buf->data);
    delete buf;
  }
}

size_t RocmAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

void RocmAllocator::register_this_thread() {
  std::lock_guard lock(worker_mutex_);
  allowed_threads_.insert(std::this_thread::get_id());
}

void RocmAllocator::rocm_free(void* buf) {
  // If rocm_free() is called from a unregistered thread, reschedule the call to
  // worker.
  {
    std::lock_guard lock(worker_mutex_);
    if (allowed_threads_.count(std::this_thread::get_id()) == 0) {
      if (!worker_) {
        worker_.reset(new Worker);
      }
      worker_->add_task([this, buf]() { this->rocm_free(buf); });
      worker_->end_batch();
      worker_->commit();
      return;
    }
  }

  hipFree(buf);
}

size_t RocmAllocator::get_active_memory() const {
  return active_memory_;
}

size_t RocmAllocator::get_peak_memory() const {
  return peak_memory_;
}

void RocmAllocator::reset_peak_memory() {
  std::lock_guard lock(mutex_);
  peak_memory_ = 0;
}

size_t RocmAllocator::get_memory_limit() {
  return memory_limit_;
}

size_t RocmAllocator::set_memory_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  std::swap(limit, memory_limit_);
  return limit;
}

size_t RocmAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t RocmAllocator::set_cache_limit(size_t limit) {
  std::lock_guard lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
}

void RocmAllocator::clear_cache() {
  std::lock_guard lk(mutex_);
  buffer_cache_.clear();
}

RocmAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of RocmAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static RocmAllocator* allocator_ = new RocmAllocator;
  return *allocator_;
}

} // namespace rocm

namespace allocator {

Allocator& allocator() {
  return rocm::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<rocm::RocmBuffer*>(ptr_)->data;
}

} // namespace allocator

size_t get_active_memory() {
  return rocm::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return rocm::allocator().get_peak_memory();
}
void reset_peak_memory() {
  return rocm::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return rocm::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return rocm::allocator().get_memory_limit();
}
size_t get_cache_memory() {
  return rocm::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return rocm::allocator().set_cache_limit(limit);
}
void clear_cache() {
  rocm::allocator().clear_cache();
}

// Not supported in ROCm.
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace mlx::core