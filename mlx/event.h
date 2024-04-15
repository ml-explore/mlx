// Copyright © 2024 Apple Inc.
#pragma once

#include <memory>
#include <stdexcept>

#include "mlx/stream.h"

namespace mlx::core {

class Event {
 public:
  Event(){};

  Event(const Stream& steam);

  // Wait for the event to be signaled at it's curent value
  void wait();

  // Signal the event at it's current value
  void signal();

  // Check if the event is valid
  bool valid() {
    return event_ != nullptr;
  };

  uint64_t value() {
    return value_;
  };

  void set_value(uint64_t v) {
    value_ = v;
  };

  const Stream& stream() {
    if (!valid()) {
      throw std::runtime_error(
          "[Event::stream] Cannot access stream on invalid event.");
    }
    return stream_;
  };

  const std::shared_ptr<void>& raw_event() {
    return event_;
  };

 private:
  // Default constructed stream should never be used
  // since the event is not yet valid
  Stream stream_{0, Device::cpu};
  std::shared_ptr<void> event_{nullptr};
  uint64_t value_{0};
};

} // namespace mlx::core
