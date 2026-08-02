// Copyright © 2024 Apple Inc.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>

#include "mlx/api.h"
#include "mlx/stream.h"

namespace mlx::core {

class MLX_API Event {
 public:
  Event() {};
  explicit Event(Stream stream);
  Event(Stream stream, bool enable_timing);

  // Record the event in the given stream
  void record(Stream stream);

  // Return the elapsed time to the end event in milliseconds
  double elapsed_time(const Event& end_event) const;

  // Check if the event has completed. Unrecorded events are complete.
  bool query() const {
    return !recorded_ || is_signaled();
  }

  // Block the CPU until the event completes
  void synchronize() {
    wait();
  }

  // Wait for the event to be signaled at its current value
  void wait();

  // Wait in the given stream for the event to be signaled at its current value
  void wait(Stream stream);

  // Signal the event at its current value in the given stream
  void signal(Stream stream);

  // Check if the event has been signaled at its current value
  bool is_signaled() const;

  // Check if the event is valid
  bool valid() const {
    return event_ != nullptr;
  }

  uint64_t value() const {
    return value_;
  }

  void set_value(uint64_t v) {
    value_ = v;
  }

  const Stream& stream() const {
    if (!valid()) {
      throw std::runtime_error(
          "[Event::stream] Cannot access stream on invalid event.");
    }
    return stream_;
  }

 private:
  // Default constructed stream should never be used
  // since the event is not yet valid
  Stream stream_{0, Device::cpu};
  std::shared_ptr<void> event_{nullptr};
  uint64_t value_{0};
  bool enable_timing_{false};
  bool recorded_{false};
};

} // namespace mlx::core
