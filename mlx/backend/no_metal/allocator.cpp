// Copyright © 2023 Apple Inc.

#include "mlx/allocator.h"

namespace mlx::core {

namespace allocator {

Allocator& allocator() {
  static CommonAllocator allocator_;
  return allocator_;
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  return static_cast<size_t*>(ptr_) + 1;
}
} // namespace allocator

size_t get_active_memory() {
  return 0;
}
size_t get_peak_memory() {
  return 0;
}
void reset_peak_memory() {}
size_t get_cache_memory() {
  return 0;
}
size_t set_memory_limit(size_t) {
  return 0;
}
size_t get_memory_limit() {
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
