// Copyright © 2023 Apple Inc.

#pragma once

#include <cstdlib>
#include <functional>
#include <utility>

#include "mlx/api.h"

namespace mlx::core::cu {
class CudaAllocator;
} // namespace mlx::core::cu

namespace mlx::core::allocator {

// Simple wrapper around buffer pointers
// WARNING: Only Buffer objects constructed from and those that wrap
//          raw pointers from mlx::allocator are supported.
class MLX_API Buffer {
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

class MLX_API Allocator {
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

MLX_API Allocator& allocator();

using Deleter = std::function<void(Buffer)>;

// A null deleter means the allocator owns the buffer
class Data {
 public:
  Buffer buffer;
  Deleter d;

  Data(Buffer buffer, Deleter d)
      : buffer(buffer), d(d ? std::move(d) : Deleter([](Buffer) {})) {}
  Data(const Data& other) = delete;
  Data& operator=(const Data& other) = delete;
  Data(Data&& other) noexcept
      : buffer(std::exchange(other.buffer, Buffer(nullptr))),
        d(std::exchange(other.d, nullptr)) {}
  ~Data() {
    if (d) {
      d(buffer);
    } else if (buffer.ptr()) {
      allocator().free(buffer);
    }
  }

 private:
  explicit Data(Buffer buffer) : buffer(buffer) {}
  friend Data malloc(size_t size);
  friend class cu::CudaAllocator;
};

inline Data malloc(size_t size) {
  return Data(allocator().malloc(size));
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

MLX_API bool can_reuse_alien_buffer(void* ptr);

} // namespace mlx::core::allocator
