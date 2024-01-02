// Copyright © 2023 Apple Inc.

#include <cstdlib>
#include <sstream>

#include "mlx/allocator.h"
#include "mlx/scheduler.h"

namespace mlx::core::allocator {

Buffer malloc(size_t size) {
  auto buffer = allocator().malloc(size, /* allow_swap */ true);
  if (size && !buffer.ptr()) {
    std::ostringstream msg;
    msg << "[malloc] Unable to allocate " << size << " bytes.";
    throw std::runtime_error(msg.str());
  }
  return buffer;
}

void free(Buffer buffer) {
  return allocator().free(buffer);
}

Buffer CommonAllocator::malloc(size_t size, bool) {
  return Buffer{std::malloc(size)};
}

void CommonAllocator::free(Buffer buffer) {
  std::free(buffer.raw_ptr());
}

Buffer malloc_or_wait(size_t size) {
  auto buffer = allocator().malloc(size);

  while (size && !buffer.ptr() && scheduler::n_active_tasks() > 0) {
    scheduler::wait_for_one();
    buffer = allocator().malloc(size);
  }

  // Try swapping if needed
  if (size && !buffer.ptr()) {
    buffer = allocator().malloc(size, /* allow_swap = */ true);
  }

  if (size && !buffer.ptr()) {
    std::ostringstream msg;
    msg << "[malloc_or_wait] Unable to allocate " << size << " bytes.";
    throw std::runtime_error(msg.str());
  }

  return buffer;
}

} // namespace mlx::core::allocator
