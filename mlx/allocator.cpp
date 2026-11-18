// Copyright © 2026 Apple Inc.

#include <algorithm>
#include <mutex>

#include "mlx/allocator.h"

namespace mlx::core::allocator {

#ifndef NDEBUG

bool Allocator::has_invalid_memory(const std::vector<void*>& ptrs) {
  std::shared_lock lock(allocations_mutex_);
  return std::any_of(ptrs.begin(), ptrs.end(), [this](void* ptr) {
    if (!ptr) {
      return false;
    }
    auto it = allocations_.upper_bound(ptr);
    if (it == allocations_.begin()) {
      return true;
    }
    --it;
    auto offset = static_cast<char*>(ptr) - static_cast<char*>(it->first);
    return static_cast<size_t>(offset) > it->second;
  });
}

void Allocator::track_allocation(void* data_ptr, size_t size) {
  if (data_ptr && size > 0) {
    std::unique_lock lock(allocations_mutex_);
    allocations_[data_ptr] = size;
  }
}

void Allocator::untrack_allocation(void* data_ptr) {
  if (data_ptr) {
    std::unique_lock lock(allocations_mutex_);
    allocations_.erase(data_ptr);
  }
}

void Allocator::swap_allocation(void* data_ptr_a, void* data_ptr_b) {
  std::unique_lock lock(allocations_mutex_);
  auto it = allocations_.find(data_ptr_a);
  if (it != allocations_.end()) {
    allocations_[data_ptr_b] = it->second;
    allocations_.erase(it);
  }
}

#endif // NDEBUG

} // namespace mlx::core::allocator
