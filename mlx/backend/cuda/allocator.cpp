// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/cuda/worker.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <cassert>

namespace mlx::core {

namespace cu {

CudaAllocator::CudaAllocator() {
  // TODO: Set memory limit for multi-device.
  size_t free, total;
  CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
  memory_limit_ = total * 0.8;
}

Buffer CudaAllocator::malloc(size_t size) {
  // TODO: Check memory limit.
  auto* buf = new CudaBuffer{nullptr, size};
  cudaError_t err = cudaMallocManaged(&buf->data, size);
  if (err != cudaSuccess && err != cudaErrorMemoryAllocation) {
    throw std::runtime_error(
        fmt::format("cudaMallocManaged failed: {}.", cudaGetErrorString(err)));
  }
  std::lock_guard lock(mutex_);
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{buf};
}

void CudaAllocator::free(Buffer buffer) {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }

  // If free() is called from a unregistered thread, reschedule the call to
  // worker.
  {
    std::lock_guard lock(worker_mutex_);
    if (allowed_threads_.count(std::this_thread::get_id()) == 0) {
      if (!worker_) {
        worker_.reset(new Worker);
      }
      worker_->add_task([buffer]() { allocator().free(buffer); });
      worker_->end_batch();
      worker_->commit();
      return;
    }
  }

  size_t size = buf->size;
  cudaFree(buf->data);
  delete buf;
  std::lock_guard lock(mutex_);
  active_memory_ -= size;
}

size_t CudaAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

void CudaAllocator::register_this_thread() {
  std::lock_guard lock(worker_mutex_);
  allowed_threads_.insert(std::this_thread::get_id());
}

size_t CudaAllocator::get_active_memory() const {
  return active_memory_;
}

size_t CudaAllocator::get_peak_memory() const {
  return peak_memory_;
}

void CudaAllocator::reset_peak_memory() {
  std::lock_guard lock(mutex_);
  peak_memory_ = 0;
}

size_t CudaAllocator::get_memory_limit() {
  return memory_limit_;
}

size_t CudaAllocator::set_memory_limit(size_t limit) {
  std::lock_guard lock(mutex_);
  std::swap(limit, memory_limit_);
  return limit;
}

CudaAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of CudaAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static CudaAllocator* allocator_ = new CudaAllocator;
  return *allocator_;
}

} // namespace cu

namespace allocator {

Allocator& allocator() {
  return cu::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<cu::CudaBuffer*>(ptr_)->data;
}

} // namespace allocator

size_t get_active_memory() {
  return cu::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return cu::allocator().get_peak_memory();
}
void reset_peak_memory() {
  return cu::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return cu::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return cu::allocator().get_memory_limit();
}

// TODO: Implement buffer cache.
size_t get_cache_memory() {
  return 0;
}
size_t set_cache_limit(size_t) {
  return 0;
}
size_t set_wired_limit(size_t) {
  return 0;
}
void clear_cache() {}

} // namespace mlx::core
