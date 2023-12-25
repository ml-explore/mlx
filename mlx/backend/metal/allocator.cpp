// Copyright © 2023 Apple Inc.

#include <iostream>
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

MetalAllocator::MetalAllocator()
    : device_(device(mlx::core::Device::gpu).mtl_device()),
      peak_allocated_size_(0),
      block_limit_(device_->recommendedMaxWorkingSetSize()) {}

Buffer MetalAllocator::malloc(size_t size, bool allow_swap /* = false */) {
  // Align up memory
  ///if (size > vm_page_size) {
  //  size = vm_page_size * ((size + vm_page_size - 1) / vm_page_size);
  //}

//  MTL::Buffer* buf = buffer_cache_.reuse_from_cache(size);

  // Prepare to allocate new memory as needed
//  if (!buf) {
    // If we have memory pressure, first check if we can reclaim some memory
    // from the cache
//    if (auto new_size = device_->currentAllocatedSize() + size; new_size >= block_limit_) {
//      buffer_cache_.clear();
//      buffer_cache_.release_cached_buffers(
//          std::max(new_size - block_limit_, size));
//    }

    // If there is still too much memory pressure, fail (likely causes a wait).
    // size + allocated (to avoid going over the limit)
    if (!allow_swap && device_->currentAllocatedSize() + size >= block_limit_) {
      return Buffer{nullptr};
    }
//  }

    // Allocate new buffer if needed
    size_t res_opt = MTL::ResourceStorageModeShared;
    res_opt |= MTL::ResourceHazardTrackingModeTracked;
    auto buf = device_->newBuffer(size, res_opt);

  peak_allocated_size_ =
      std::max(peak_allocated_size_, device_->currentAllocatedSize());

  return Buffer{static_cast<void*>(buf)};
}

void MetalAllocator::free(Buffer buffer) {
  static_cast<MTL::Buffer*>(buffer.ptr())->release();
}

MetalAllocator& allocator() {
  static MetalAllocator allocator_;
  return allocator_;
}

} // namespace metal

} // namespace mlx::core
