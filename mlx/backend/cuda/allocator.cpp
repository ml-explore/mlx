// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/utils.h"

#include <cuda_runtime.h>
#include <fmt/format.h>

namespace mlx::core {

namespace mxcuda {

CudaAllocator::CudaAllocator() {
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
        fmt::format("cudaMallocManaged failed: {}", cudaGetErrorString(err)));
  }
  std::unique_lock lk(mutex_);
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);
  return Buffer{buf};
}

void CudaAllocator::free(Buffer buffer) {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return;
  }
  size_t size = buf->size;
  cudaFree(buf->data);
  delete buf;
  std::unique_lock lk(mutex_);
  active_memory_ -= size;
}

size_t CudaAllocator::size(Buffer buffer) const {
  auto* buf = static_cast<CudaBuffer*>(buffer.ptr());
  if (!buf) {
    return 0;
  }
  return static_cast<CudaBuffer*>(buffer.ptr())->size;
}

CudaAllocator& allocator() {
  // By creating the |allocator_| on heap, the destructor of CudaAllocator
  // will not be called on exit and buffers in the cache will be leaked. This
  // can save some time at program exit.
  static CudaAllocator* allocator_ = new CudaAllocator;
  return *allocator_;
}

} // namespace mxcuda

namespace allocator {

Allocator& allocator() {
  return mxcuda::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<mxcuda::CudaBuffer*>(ptr_)->data;
}

} // namespace allocator

size_t get_active_memory() {
  return mxcuda::allocator().get_active_memory();
}
size_t get_peak_memory() {
  return mxcuda::allocator().get_peak_memory();
}
void reset_peak_memory() {
  return mxcuda::allocator().reset_peak_memory();
}
size_t set_memory_limit(size_t limit) {
  return mxcuda::allocator().set_memory_limit(limit);
}
size_t get_memory_limit() {
  return mxcuda::allocator().get_memory_limit();
}

// No-ops for common allocator
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
