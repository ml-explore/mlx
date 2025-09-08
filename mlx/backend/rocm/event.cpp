// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/event.h"
#include "mlx/backend/rocm/utils.h"

namespace mlx::core::rocm {

HipEvent::HipEvent() {
  CHECK_HIP_ERROR(hipEventCreate(&event_));
}

HipEvent::~HipEvent() {
  CHECK_HIP_ERROR(hipEventDestroy(event_));
}

void HipEvent::record(hipStream_t stream) {
  CHECK_HIP_ERROR(hipEventRecord(event_, stream));
}

void HipEvent::wait() {
  CHECK_HIP_ERROR(hipEventSynchronize(event_));
}

bool HipEvent::query() const {
  hipError_t status = hipEventQuery(event_);
  if (status == hipSuccess) {
    return true;
  } else if (status == hipErrorNotReady) {
    return false;
  } else {
    CHECK_HIP_ERROR(status);
    return false;
  }
}

SharedEvent::SharedEvent() = default;

void SharedEvent::notify() {
  std::lock_guard<std::mutex> lock(mutex_);
  ready_ = true;
  cv_.notify_one();
}

void SharedEvent::wait() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this] { return ready_; });
  ready_ = false;
}

} // namespace mlx::core::rocm