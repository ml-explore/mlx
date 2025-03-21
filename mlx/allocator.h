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
  Buffer(void* ptr) : ptr_(ptr) {};

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

Buffer malloc(size_t size);

void free(Buffer buffer);

class Allocator {
  /** Abstract base class for a memory allocator. */
 public:
  virtual Buffer malloc(size_t size) = 0;
  virtual void free(Buffer buffer) = 0;
  virtual size_t size(Buffer buffer) const = 0;

  Allocator() = default;
  Allocator(const Allocator& other) = delete;
  Allocator(Allocator&& other) = delete;
  Allocator& operator=(const Allocator& other) = delete;
  Allocator& operator=(Allocator&& other) = delete;
  virtual ~Allocator() = default;
};

Allocator& allocator();

} // namespace mlx::core::allocator
