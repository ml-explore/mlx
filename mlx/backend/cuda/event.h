// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/stream.h"

#include <memory>

#include <cuda_runtime.h>
#include <cuda/atomic>

namespace mlx::core::cu {

class Device;

// RAII-managed move-only wrapper of cudaEvent_t.
struct CudaEventHandle : public CudaHandle<cudaEvent_t, cudaEventDestroy> {
  CudaEventHandle(Device& d, int flags);
  Device& device;
  int flags;
};

// Wrapper of native cuda event. It can synchronize between GPU streams, or wait
// on GPU stream in CPU stream, but can not wait on CPU stream.
class CudaEvent {
 public:
  CudaEvent(Device& d, int flags);
  ~CudaEvent();

  CudaEvent(CudaEvent&&) = default;
  CudaEvent& operator=(CudaEvent&&) = default;

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  void wait();
  void wait(cudaStream_t stream);
  void record(cudaStream_t stream);

  // Return whether the recorded kernels have completed. Note that this method
  // returns true if record() has not been called.
  bool completed() const;

  // Internal: make sure event pool is initialized.
  static void init_pool();

 private:
  CudaEventHandle event_;
};

// Event that can synchronize between CPU and GPU. It is much slower than
// CudaEvent so the latter should always be preferred when possible.
class AtomicEvent {
 public:
  AtomicEvent(Device& d);

  void wait(uint32_t value);
  void wait(cudaStream_t stream, uint32_t value);
  void wait(Stream s, uint32_t value);
  void signal(uint32_t value);
  void signal(cudaStream_t stream, uint32_t value);
  void signal(Stream s, uint32_t value);
  bool is_signaled(uint32_t value) const;
  uint32_t value() const;

 private:
  const CudaStream& signal_stream();

  uint32_t* ptr() const {
    return static_cast<uint32_t*>(buf_.get());
  }

  bool coherent_;
  std::shared_ptr<void> buf_;
};

} // namespace mlx::core::cu
