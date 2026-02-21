// Copyright © 2026 Apple Inc.

#include <atomic>
#include <cstdint>

#include "mlx/allocator.h"
#include "mlx/backend/gpu/failure.h"
#include "mlx/backend/metal/failure.h"
#include "mlx/failure.h"

namespace mlx::core {

namespace {

class Failure {
 public:
  static Failure& get() {
    static Failure instance;
    return instance;
  }

  Failure(const Failure&) = delete;
  Failure& operator=(const Failure&) = delete;

  allocator::Buffer& buffer() {
    return buffer_;
  }

  void reset() {
    atomic_ptr()->store(FailureCode::NoFailure, std::memory_order_relaxed);
  }

  FailureCode value() {
    return atomic_ptr()->load(std::memory_order_relaxed);
  }

 private:
  Failure() : buffer_(allocator::malloc(sizeof(int32_t))) {
    reset();
  }
  ~Failure() = default;

  std::atomic<FailureCode>* atomic_ptr() {
    return reinterpret_cast<std::atomic<FailureCode>*>(buffer_.raw_ptr());
  }

  allocator::Buffer buffer_;
};

} // namespace

namespace gpu {

void reset_failure() {
  Failure::get().reset();
}

bool has_failure() {
  return Failure::get().value() != FailureCode::NoFailure;
}

} // namespace gpu

namespace metal {

MTL::Buffer* get_failure_buffer() {
  return static_cast<MTL::Buffer*>(Failure::get().buffer().ptr());
}

} // namespace metal

} // namespace mlx::core
