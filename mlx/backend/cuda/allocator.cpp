// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/utils.h"

#include <cuda_runtime.h>
#include <fmt/format.h>
#include <unistd.h>

#include <cassert>

namespace mlx::core {

namespace cu {

constexpr int page_size = 16384;

// Any allocations smaller than this will try to use the small pool
constexpr int small_block_size = 8;

// The small pool size in bytes. This should be a multiple of the host page
// size and small_block_size.
constexpr int small_pool_size = 4 * page_size;

SmallSizePool::SmallSizePool() {
  CHECK_CUDA_ERROR(cudaMallocManaged(&buffer_, small_pool_size));
  end_ = reinterpret_cast<void*>(
      reinterpret_cast<char*>(buffer_) + small_pool_size);
  next_free_ = reinterpret_cast<Block*>(buffer_);

  auto num_blocks = small_pool_size / small_block_size;
  auto curr = next_free_;
  for (size_t i = 0; i < num_blocks - 1; ++i) {
    curr->next = reinterpret_cast<Block*>(
        reinterpret_cast<char*>(buffer_) + (i + 1) * small_block_size);
    curr = curr->next;
  }
  curr->next = nullptr;
}

SmallSizePool::~SmallSizePool() {
  CHECK_CUDA_ERROR(cudaFree(buffer_));
}

void* SmallSizePool::malloc() {
  if (next_free_ == nullptr) {
    return nullptr;
  }
  Block* b = next_free_;
  next_free_ = next_free_->next;
  return static_cast<void*>(b);
}

void SmallSizePool::free(void* p) {
  auto b = static_cast<Block*>(p);
  b->next = next_free_;
  next_free_ = b;
}

bool SmallSizePool::in_pool(void* p) {
  return (p >= buffer_) && (p < end_);
}

CudaAllocator::CudaAllocator()
    : buffer_cache_(
          page_size,
          [](CudaBuffer* buf) { return buf->size; },
          [this](CudaBuffer* buf) {
            cuda_free(buf->data);
            delete buf;
          }) {
  // TODO: Set memory limit for multi-device.
  size_t free, total;
  CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total));
  memory_limit_ = total * 0.8;
  max_pool_size_ = memory_limit_;
}

Buffer CudaAllocator::malloc(size_t size) {
  // Find available buffer from cache.
  auto orig_size = size;
  std::unique_lock lock(mutex_);
  if (size <= small_block_size) {
    size = 8;
  } else if (size < page_size) {
    size = next_power_of_2(size);
  } else {
    size = page_size * ((size + page_size - 1) / page_size);
  }

  CudaBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    // If we have a lot of memory pressure or are over the maximum cache size,
    // try to reclaim memory from the cache.
    size_t mem_required = get_active_memory() + get_cache_memory() + size;
    if (mem_required >= memory_limit_) {
      buffer_cache_.release_cached_buffers(mem_required - memory_limit_);
    }

    lock.unlock();
    buf = new CudaBuffer{nullptr, size};

    // Try the scalar pool first
    if (size <= small_block_size) {
      buf->data = scalar_pool_.malloc();
    }
    if (!buf->data) {
      cudaError_t err = cudaMallocManaged(&buf->data, size);
      if (err != cudaSuccess && err != cudaErrorMemoryAllocation) {
        throw std::runtime_error(fmt::format(
            "cudaMallocManaged failed: {}.", cudaGetErrorString(err)));
      }
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

void CudaAllocator::free(Buffer buffer) {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    lock.unlock();
    cuda_free(buf->data);
    delete buf;
  }
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

void CudaAllocator::cuda_free(void* buf) {
  // If cuda_free() is called from a unregistered thread, reschedule the call to
  // worker.
  {
    std::lock_guard lock(worker_mutex_);
    if (allowed_threads_.count(std::this_thread::get_id()) == 0) {
      if (!worker_) {
        worker_.reset(new Worker);
      }
      worker_->add_task([this, buf]() { this->cuda_free(buf); });
      worker_->end_batch();
      worker_->commit();
      return;
    }
  }
  if (scalar_pool_.in_pool(buf)) {
    scalar_pool_.free(buf);
  } else {
    cudaFree(buf);
  }
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

size_t CudaAllocator::get_cache_memory() const {
  return buffer_cache_.cache_size();
}

size_t CudaAllocator::set_cache_limit(size_t limit) {
  std::lock_guard lk(mutex_);
  std::swap(limit, max_pool_size_);
  return limit;
}

void CudaAllocator::clear_cache() {
  std::lock_guard lk(mutex_);
  buffer_cache_.clear();
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
size_t get_cache_memory() {
  return cu::allocator().get_cache_memory();
}
size_t set_cache_limit(size_t limit) {
  return cu::allocator().set_cache_limit(limit);
}
void clear_cache() {
  cu::allocator().clear_cache();
}

// Not supported in CUDA.
size_t set_wired_limit(size_t) {
  return 0;
}

} // namespace mlx::core
