// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/stream.h"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <memory>

namespace mlx::core::cu {

// Event that can synchronize between CPU and GPU. It is much slower than
// CudaEvent so the latter should always be preferred when possible.
class SharedEvent {
 public:
  using Atomic = cuda::atomic<uint64_t>;

  SharedEvent();

  void wait(uint64_t value);
  void wait(cudaStream_t stream, uint64_t value);
  void wait(Stream s, uint64_t value);
  void signal(uint64_t value);
  void signal(cudaStream_t stream, uint64_t value);
  void signal(Stream s, uint64_t value);
  bool is_signaled(uint64_t value) const;
  uint64_t value() const;

 private:
  std::shared_ptr<mlx::core::allocator::Buffer> buf_;
};

} // namespace mlx::core::cu
