// Copyright © 2023 Apple Inc.

#pragma once

#include <cstdlib>

namespace mlx::core::allocator {

// Simple wrapper around buffer pointers
// WARNING: Only Buffer objects constructed from and those that wrap
//          raw pointers from mlx::allocator are supported.
class Buffer {
 private:
  void* ptr_;

 public:
  explicit Buffer(void* ptr) : ptr_(ptr) {};

  // Get the raw data pointer from the buffer
  void* raw_ptr();

  // Get the buffer pointer from the buffer
  const void* ptr() const {
    return ptr_;
  };
  void* ptr() {
    return ptr_;
  };
};

class Allocator {
  /** Abstract base class for a memory allocator. */
 public:
  virtual Buffer malloc(size_t size) = 0;
  virtual void free(Buffer buffer) = 0;
  virtual size_t size(Buffer buffer) const = 0;
  virtual Buffer make_buffer(void* ptr, size_t size) {
    return Buffer{nullptr};
  };
  virtual void release(Buffer buffer) {}

  Allocator() = default;
  Allocator(const Allocator& other) = delete;
  Allocator(Allocator&& other) = delete;
  Allocator& operator=(const Allocator& other) = delete;
  Allocator& operator=(Allocator&& other) = delete;
  virtual ~Allocator() = default;
};

Allocator& allocator();

inline Buffer malloc(size_t size) {
  return allocator().malloc(size);
}

inline void free(Buffer buffer) {
  allocator().free(buffer);
}

// Make a Buffer from a raw pointer of the given size without a copy.  If a
// no-copy conversion is not possible then the returned buffer.ptr() will be
// nullptr. Any buffer created with this function must be released with
// release(buffer)
inline Buffer make_buffer(void* ptr, size_t size) {
  return allocator().make_buffer(ptr, size);
};

// Release a buffer from the allocator made with make_buffer
inline void release(Buffer buffer) {
  allocator().release(buffer);
}

} // namespace mlx::core::allocator
