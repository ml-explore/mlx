// Copyright © 2023 Apple Inc.

#include "mlx/allocator.h"

namespace mlx::core::allocator {

Allocator& allocator() {
  static CommonAllocator allocator_;
  return allocator_;
}

void* Buffer::raw_ptr() {
  return static_cast<size_t*>(ptr_) + 1;
}

} // namespace mlx::core::allocator
