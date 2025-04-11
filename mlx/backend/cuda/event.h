// Copyright © 2025 Apple Inc.

#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>

namespace mlx::core::mxcuda {

// Event that can synchronize between CPU and GPU.
class SharedEvent {
 public:
  // The cuda::atomic can synchronize between CPU and GPU, with a little
  // performance penalty: some ops can take more than 1ms.
  using Atomic = cuda::atomic<uint64_t>;

  SharedEvent();

  void wait(uint64_t value);
  void wait(Stream stream, uint64_t value);
  void signal(uint64_t value);
  void signal(Stream stream, uint64_t value);
  bool is_signaled(uint64_t value) const;

 private:
  std::shared_ptr<Atomic> ac_;
};

// Wrapper of CUDA events.
class CudaEvent {
 public:
  CudaEvent();
  ~CudaEvent();

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  void wait();
  void wait(Stream stream);
  void record(Stream stream);
  bool completed() const;

 private:
  cudaEvent_t event_;
};

} // namespace mlx::core::mxcuda
