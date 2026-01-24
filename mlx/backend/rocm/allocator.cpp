// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/utils.h"

#include <hip/hip_runtime.h>
#include <fmt/format.h>
#include <unistd.h>

#include <cassert>

namespace mlx::core {

namespace rocm {

constexpr int page_size = 16384;

// Any allocations smaller than this will try to use the small pool
constexpr int small_block_size = 8;

// The small pool size in bytes. This should be a multiple of the host page
// size and small_block_size.
constexpr int small_pool_size = 4 * page_size;

SmallSizePool::SmallSizePool() {
  auto num_blocks = small_pool_size / small_block_size;
  buffer_ = new Block[num_blocks];

  next_free_ = buffer_;

  CHECK_HIP_ERROR(hipMallocManaged(&data_, small_pool_size));
  CHECK_HIP_ERROR(
      hipMemAdvise(data_, small_pool_size, hipMemAdviseSetReadMostly, 0));

  auto curr = next_free_;
  for (size_t i = 1; i < num_blocks; ++i) {
    curr->next = buffer_ + i;
    curr = curr->next;
  }
  curr->next = nullptr;
}

SmallSizePool::~SmallSizePool() {
  CHECK_HIP_ERROR(hipFree(data_));
  delete[] buffer_;
}

RocmBuffer* SmallSizePool::malloc() {
  if (next_free_ == nullptr) {
    return nullptr;
  }
  Block* b = next_free_;
  uint64_t i = next_free_ - buffer_;
  next_free_ = next_free_->next;
  b->buf.data = static_cast<char*>(data_) + i * small_block_size;
  b->buf.size = small_block_size;
  return &b->buf;
}

void SmallSizePool::free(RocmBuffer* buf) {
  auto b = reinterpret_cast<Block*>(buf);
  b->next = next_free_;
  next_free_ = b;
}

bool SmallSizePool::in_pool(RocmBuffer* buf) {
  constexpr int num_blocks = (small_pool_size / small_block_size);
  auto b = reinterpret_cast<Block*>(buf);
  int64_t block_num = b - buffer_;
  return block_num >= 0 && block_num < num_blocks;
}

RocmAllocator::RocmAllocator()
    : buffer_cache_(
          page_size,
          [](RocmBuffer* buf) { return buf->size; },
          [this](RocmBuffer* buf) { rocm_free(buf); }) {
  // TODO: Set memory limit for multi-device.
  size_t free, total;
  CHECK_HIP_ERROR(hipMemGetInfo(&free, &total));
  memory_limit_ = total * 0.8;
  max_pool_size_ = memory_limit_;
}

Buffer RocmAllocator::malloc(size_t size) {
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

  RocmBuffer* buf = buffer_cache_.reuse_from_cache(size);
  if (!buf) {
    // If we have a lot of memory pressure try to reclaim memory from the cache.
    int64_t mem_to_free =
        get_active_memory() + get_cache_memory() + size - memory_limit_;
    if (mem_to_free > 0) {
      buffer_cache_.release_cached_buffers(mem_to_free);
    }

    // Try the scalar pool first
    if (size <= small_block_size) {
      buf = scalar_pool_.malloc();
    }
    lock.unlock();
    if (!buf) {
      buf = new RocmBuffer{nullptr, size};
      hipError_t err = hipMallocManaged(&buf->data, size);
      if (err != hipSuccess && err != hipErrorMemoryAllocation) {
        throw std::runtime_error(fmt::format(
            "hipMallocManaged failed: {}.", hipGetErrorString(err)));
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
    rocm_free(buf);
  }
}

size_t RocmAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<RocmBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

// This must be called with mutex_ acquired
void RocmAllocator::rocm_free(RocmBuffer* buf) {
  if (scalar_pool_.in_pool(buf)) {
    scalar_pool_.free(buf);
  } else {
    hipFree(buf->data);
    delete buf;
  }
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
