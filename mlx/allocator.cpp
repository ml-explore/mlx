// Copyright © 2023 Apple Inc.

#include <cstdlib>
#include <sstream>

#include "mlx/allocator.h"

namespace mlx::core::allocator {

Buffer malloc(size_t size) {
  auto buffer = allocator().malloc(size);
  if (size && !buffer.ptr()) {
    std::ostringstream msg;
    msg << "[malloc] Unable to allocate " << size << " bytes.";
    throw std::runtime_error(msg.str());
  }
  return buffer;
}

void free(Buffer buffer) {
  allocator().free(buffer);
}

} // namespace mlx::core::allocator
