// Copyright © 2025 Apple Inc.

#include <hip/hip_runtime.h>
#include "mlx/backend/rocm/utils.h"

namespace mlx::core::rocm {

class Event {
public:
  Event() {
    check_hip_error("hipEventCreate", hipEventCreate(&event_));
  }
  
  ~Event() {
    hipEventDestroy(event_);
  }
  
  void record(hipStream_t stream) {
    check_hip_error("hipEventRecord", hipEventRecord(event_, stream));
  }
  
  void wait() {
    check_hip_error("hipEventSynchronize", hipEventSynchronize(event_));
  }
  
  hipEvent_t event() const { return event_; }

private:
  hipEvent_t event_;
};

} // namespace mlx::core::rocm 