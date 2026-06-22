// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <condition_variable>
#include <mutex>

namespace mlx::core {

struct EventCounter {
  uint64_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
  Event::Error error;

  void wait(uint64_t val) {
    std::unique_lock<std::mutex> lk(mtx);
    if (value >= val) {
      return;
    }
    cv.wait(lk, [this, val] { return value >= val; });
  }
};

Event::Event(Stream stream) : stream_(stream) {
  event_ = std::make_shared<EventCounter>();
}

void Event::wait() {
  check_error();
  cast<EventCounter>().wait(value());
  check_error();
}

void Event::wait(Stream stream) {
  scheduler::wait_event(stream, *this, [value = value()](Event& self) {
    self.cast<EventCounter>().wait(value);
  });
}

void Event::signal(Stream stream) {
  scheduler::signal_event(stream, *this, [value = value()](Event& self) {
    auto& ec = self.cast<EventCounter>();
    {
      std::lock_guard<std::mutex> lk(ec.mtx);
      ec.value = value;
    }
    ec.cv.notify_all();
  });
}

bool Event::is_signaled() const {
  auto& ec = cast<EventCounter>();
  std::lock_guard<std::mutex> lk(ec.mtx);
  return ec.value >= value();
}

Event::Error& Event::error() {
  return cast<EventCounter>().error;
}

} // namespace mlx::core
