// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/memory.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

#include <cassert>

namespace mlx::core {

namespace cu {

constexpr int page_size = 16384;

// Any allocations smaller than this will try to use the small pool
constexpr int small_block_size = 8;

// The small pool size in bytes. This should be a multiple of the host page
// size and small_block_size.
constexpr int small_pool_size = 4 * page_size;

bool supports_managed_memory() {
  static bool managed_memory = []() {
    int device_count = gpu::device_count();
    for (int i = 0; i < device_count; ++i) {
      auto& d = cu::device(i);
      if (!d.managed_memory()) {
        return false;
      }
#if defined(_WIN32)
      // Empirically on Windows if there is no concurrentManagedAccess the
      // managed memory also does not work.
      if (!d.concurrent_managed_access()) {
        return false;
      }
#endif
    }
    return true;
  }();
  return managed_memory;
}

inline void* unified_malloc(size_t size) {
  void* data = nullptr;
  if (supports_managed_memory()) {
    CHECK_CUDA_ERROR(cudaMallocManaged(&data, size));
  } else {
    CHECK_CUDA_ERROR(cudaMallocHost(&data, size));
  }
  return data;
}

inline void unified_free(void* data) {
  if (supports_managed_memory()) {
    CHECK_CUDA_ERROR(cudaFree(data));
  } else {
    CHECK_CUDA_ERROR(cudaFreeHost(data));
  }
}

#if CUDART_VERSION >= 13000
inline cudaMemLocation cuda_mem_loc(int i) {
  cudaMemLocation loc;
  loc.type = cudaMemLocationTypeDevice;
  loc.id = i;
  return loc;
}
#else
inline int cuda_mem_loc(int i) {
  return i;
}
#endif // CUDART_VERSION >= 13000

SmallSizePool::SmallSizePool() {
  auto num_blocks = small_pool_size / small_block_size;
  buffer_ = new Block[num_blocks];
  next_free_ = buffer_;

  data_ = unified_malloc(small_pool_size);
  if (supports_managed_memory()) {
    int device_count = gpu::device_count();
    for (int i = 0; i < device_count; ++i) {
      if (device(i).concurrent_managed_access()) {
        auto loc = cuda_mem_loc(i);
        CHECK_CUDA_ERROR(cudaMemAdvise(
            data_, small_pool_size, cudaMemAdviseSetAccessedBy, loc));
      }
    }
  }

  auto curr = next_free_;
  for (size_t i = 1; i < num_blocks; ++i) {
    curr->next = buffer_ + i;
    curr = curr->next;
  }
  curr->next = nullptr;
}

SmallSizePool::~SmallSizePool() {
  unified_free(data_);
  delete[] buffer_;
}

CudaBuffer* SmallSizePool::malloc() {
  if (next_free_ == nullptr) {
    return nullptr;
  }
  Block* b = next_free_;
  uint64_t i = next_free_ - buffer_;
  next_free_ = next_free_->next;
  b->buf.data = static_cast<char*>(data_) + i * small_block_size;
  b->buf.size = small_block_size;
  b->buf.device = -1;
  return &b->buf;
}

void SmallSizePool::free(CudaBuffer* buf) {
  auto b = reinterpret_cast<Block*>(buf);
  b->next = next_free_;
  next_free_ = b;
}

bool SmallSizePool::in_pool(CudaBuffer* buf) {
  constexpr int num_blocks = (small_pool_size / small_block_size);
  auto b = reinterpret_cast<Block*>(buf);
  int64_t block_num = b - buffer_;
  return block_num >= 0 && block_num < num_blocks;
}

CudaAllocator::CudaAllocator()
    : buffer_cache_(
          page_size,
          [](CudaBuffer* buf) { return buf->size; },
          [this](CudaBuffer* buf) { free_cuda_buffer(buf); }) {
  size_t free;
  CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total_memory_));
  memory_limit_ = total_memory_ * 0.95;
  free_limit_ = total_memory_ - memory_limit_;
  max_pool_size_ = memory_limit_;

  int device_count = gpu::device_count();
  free_streams_.resize(device_count);
  mem_pools_.resize(device_count);
  for (int i = 0; i < device_count; ++i) {
    auto& d = device(i);
    if (d.memory_pools()) {
      free_streams_[i] = CudaStream(d);
      CHECK_CUDA_ERROR(cudaDeviceGetDefaultMemPool(&mem_pools_[i], i));
    }
  }
}

Buffer
CudaAllocator::malloc_async(size_t size, int device, cudaStream_t stream) {
  if (size == 0) {
    return Buffer{new CudaBuffer{nullptr, 0, -1}};
  }

  if (size <= small_block_size) {
    size = 8;
  } else if (size < page_size) {
    size = next_power_of_2(size);
  } else {
    size = page_size * ((size + page_size - 1) / page_size);
  }

  if (size <= small_block_size || stream == nullptr) {
    device = -1;
  }

  // Find available buffer from cache.
  std::unique_lock lock(mutex_);
  CudaBuffer* buf = buffer_cache_.reuse_from_cache(size);
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
      void* data = nullptr;
      if (device == -1) {
        data = unified_malloc(size);
      } else {
        if (mem_pools_[device]) { // supports memory pools
          CHECK_CUDA_ERROR(cudaMallocAsync(&data, size, stream));
        } else {
          CHECK_CUDA_ERROR(cudaMalloc(&data, size));
        }
      }
      if (!data) {
        std::ostringstream msg;
        msg << "[malloc] Unable to allocate " << size << " bytes.";
        throw std::runtime_error(msg.str());
      }
      buf = new CudaBuffer{data, size, device};
    }
    lock.lock();

    // If any cuda memory pool has too much reserved memory, clear some
    // memory from the cache. This prevents graph / kernel execution failing
    // from OOM
    if (get_cache_memory() > 0) {
      for (auto p : mem_pools_) {
        if (p) {
          size_t used = 0;
          CHECK_CUDA_ERROR(cudaMemPoolGetAttribute(
              p, cudaMemPoolAttrReservedMemCurrent, &used));
          if (used > (total_memory_ - free_limit_)) {
            buffer_cache_.release_cached_buffers(free_limit_);
            break;
          }
        }
      }
    }
  }
  active_memory_ += buf->size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Maintain the cache below the requested limit.
  if (get_cache_memory() > max_pool_size_) {
    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);
  }
  lock.unlock();
  // Copy to unified memory here if the buffer is not on the right device.
  if (buf->device >= 0 && buf->device != device) {
    move_to_unified_memory(*buf, stream);
  }
  return Buffer{buf};
}

Buffer CudaAllocator::malloc(size_t size) {
  return malloc_async(size, -1, nullptr);
}

void CudaAllocator::free(Buffer buffer) {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  if (buf->size == 0) {
    delete buf;
    return;
  }

  std::unique_lock lock(mutex_);
  active_memory_ -= buf->size;
  if (get_cache_memory() < max_pool_size_) {
    buffer_cache_.recycle_to_cache(buf);
  } else {
    free_cuda_buffer(buf);
  }
}

size_t CudaAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return buf->size;
}

void CudaAllocator::move_to_unified_memory(
    CudaBuffer& buf,
    cudaStream_t stream) {
  if (buf.device == -1) {
    return;
  }
  void* data = unified_malloc(buf.size);
  cudaMemcpyKind kind =
      supports_managed_memory() ? cudaMemcpyDefault : cudaMemcpyDeviceToHost;
  if (stream && mem_pools_[buf.device]) {
    CHECK_CUDA_ERROR(cudaMemcpyAsync(data, buf.data, buf.size, kind, stream));
    free_async(buf, stream);
  } else {
    CHECK_CUDA_ERROR(cudaMemcpy(data, buf.data, buf.size, kind));
    free_async(buf);
  }
  buf.data = data;
  buf.device = -1;
}

// This must be called with mutex_ aquired
void CudaAllocator::free_cuda_buffer(CudaBuffer* buf) {
  if (scalar_pool_.in_pool(buf)) {
    scalar_pool_.free(buf);
  } else {
    free_async(*buf);
    delete buf;
  }
}

void CudaAllocator::free_async(CudaBuffer& buf, cudaStream_t stream) {
  if (buf.device == -1) {
    unified_free(buf.data);
  } else {
    // Free asynchronously when memory pools is supported.
    if (mem_pools_[buf.device]) {
      if (!stream) {
        stream = free_streams_[buf.device];
      }
      CHECK_CUDA_ERROR(cudaFreeAsync(buf.data, stream));
    } else {
      CHECK_CUDA_ERROR(cudaFree(buf.data));
    }
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
  static auto* allocator_ = []() {
    // Ensure scheduler is created before allocator.
    scheduler::scheduler();
    // By creating the |allocator_| on heap, the destructor of CudaAllocator
    // will not be called on exit and buffers in the cache will be leaked. This
    // can save some time at program exit.
    return new CudaAllocator();
  }();
  return *allocator_;
}

Buffer malloc_async(size_t size, CommandEncoder& encoder) {
  return allocator().malloc_async(
      size, encoder.device().cuda_device(), encoder.stream());
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
  auto& cbuf = *static_cast<cu::CudaBuffer*>(ptr_);
  cu::allocator().move_to_unified_memory(cbuf);
  return cbuf.data;
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
