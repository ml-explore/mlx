// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/stream.h"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include <memory>

namespace mlx::core::cu {

class CudaEventHandle;

// Wrapper of native cuda event. It can synchronize between GPU streams, or wait
// on GPU stream in CPU stream, but can not wait on CPU stream.
class CudaEvent {
 public:
  CudaEvent();

  void wait();
  void wait(cudaStream_t stream);
  void wait(Stream s);
  void record(cudaStream_t stream);
  void record(Stream s);

  // Return whether the recorded kernels have completed. Note that this method
  // returns true if record() has not been called.
  bool completed() const;

  bool recorded() const {
    return recorded_;
  }

 private:
  bool recorded_{false};
  std::shared_ptr<CudaEventHandle> event_;
};

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
