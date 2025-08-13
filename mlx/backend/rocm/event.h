// Copyright © 2025 Apple Inc.

#pragma once

#include <hip/hip_runtime.h>

#include <condition_variable>
#include <memory>
#include <mutex>

namespace mlx::core::rocm {

// HIP event managed with RAII.
class HipEvent {
 public:
  HipEvent();
  ~HipEvent();

  HipEvent(const HipEvent&) = delete;
  HipEvent& operator=(const HipEvent&) = delete;

  void record(hipStream_t stream);
  void wait();
  bool query() const;

  operator hipEvent_t() const {
    return event_;
  }

 private:
  hipEvent_t event_;
};

// Shared event for worker thread synchronization.
class SharedEvent {
 public:
  SharedEvent();

  void notify();
  void wait();

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  bool ready_{false};
};

} // namespace mlx::core::rocm