// Copyright © 2024 Apple Inc.
#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "mlx/stream.h"

namespace mlx::core {

class Event {
 public:
  Event() {};
  explicit Event(Stream stream);

  Event(const Event& other)
      : stream_(other.stream_),
        event_(other.event_),
        value_(other.value_.load(std::memory_order_acquire)) {}

  Event& operator=(const Event& other) {
    if (this != &other) {
      stream_ = other.stream_;
      event_ = other.event_;
      value_.store(
          other.value_.load(std::memory_order_acquire),
          std::memory_order_release);
    }
    return *this;
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
    // Acquire: readers observe state published before set_value(release).
    return value_.load(std::memory_order_acquire);
  }

  void set_value(uint64_t v) {
    // Release: publish host-side event counter updates.
    value_.store(v, std::memory_order_release);
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
  std::atomic<uint64_t> value_{0};
};

} // namespace mlx::core
