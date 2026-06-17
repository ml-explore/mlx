// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/stream.h"

#include <memory>

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

// RAII-managed move-only wrapper of hipEvent_t.
struct HipEventHandle : public HipHandle<hipEvent_t, hipEventDestroy> {
  HipEventHandle(int flags);
  int flags;
};

// Wrapper of native HIP event. It can synchronize between GPU streams, or wait
// on GPU stream in CPU stream, but can not wait on CPU stream.
class HipEvent {
 public:
  explicit HipEvent(int flags);
  ~HipEvent();

  HipEvent(HipEvent&&) = default;
  HipEvent& operator=(HipEvent&&) = default;

  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;

  void wait();
  void wait(hipStream_t stream);
  void record(hipStream_t stream);

  // Return whether the recorded kernels have completed. Note that this method
  // returns true if record() has not been called.
  bool completed() const;

 private:
  HipEventHandle event_;
};

// Event that can synchronize between CPU and GPU. It is much slower than
// HipEvent so the latter should always be preferred when possible.
class AtomicEvent {
 public:
  AtomicEvent();

  void wait(uint64_t value);
  void wait(hipStream_t stream, uint64_t value);
  void wait(Stream s, uint64_t value);
  void signal(uint64_t value);
  void signal(hipStream_t stream, uint64_t value);
  void signal(Stream s, uint64_t value);
  bool is_signaled(uint64_t value) const;
  uint64_t value() const;

 private:
  std::atomic<uint64_t>* atomic() const {
    return atomic_;
  }

  // The completion counter lives in PINNED HOST memory, not device memory. The
  // GPU writes it (hipStreamWriteValue64) and the CPU polls it (wait()). Device
  // memory — even fine-grained — is not reliably CPU-coherent on a discrete GPU
  // over a non-coherent link (e.g. an R9700 in a TB5 eGPU enclosure), so the
  // host poll would spin forever. Pinned host memory is the canonical GPU->host
  // signaling path and works on both the integrated APU and a discrete dGPU.
  std::shared_ptr<void> mem_;
  std::atomic<uint64_t>* atomic_{nullptr};
};

} // namespace mlx::core::rocm
