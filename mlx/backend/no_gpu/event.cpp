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
};

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) { delete static_cast<EventCounter*>(ptr); };
  event_ = std::shared_ptr<void>(new EventCounter{}, dtor);
}

Event::Event(Stream stream, bool enable_timing)
    : stream_(stream), enable_timing_(enable_timing) {
  throw std::invalid_argument(
      "[Event::Event] Events are only supported on GPU.");
}

void Event::record(Stream) {
  throw std::runtime_error("[Event::record] Events are only supported on GPU.");
}

double Event::elapsed_time(const Event&) const {
  throw std::runtime_error(
      "[Event::elapsed_time] Event timing is only supported on GPU.");
}

void Event::wait() {
  if (value() == 0) {
    return;
  }
  auto ec = static_cast<EventCounter*>(event_.get());
  std::unique_lock<std::mutex> lk(ec->mtx);
  if (ec->value >= value()) {
    return;
  }
  ec->cv.wait(lk, [value = value(), ec] { return ec->value >= value; });
}

void Event::wait(Stream stream) {
  if (value() == 0) {
    return;
  }
  scheduler::enqueue(stream, [*this]() mutable { wait(); });
}

void Event::signal(Stream stream) {
  scheduler::enqueue(stream, [*this]() mutable {
    auto ec = static_cast<EventCounter*>(event_.get());
    {
      std::lock_guard<std::mutex> lk(ec->mtx);
      ec->value = value();
    }
    ec->cv.notify_all();
  });
}

bool Event::is_signaled() const {
  auto ec = static_cast<EventCounter*>(event_.get());
  {
    std::lock_guard<std::mutex> lk(ec->mtx);
    return (ec->value >= value());
  }
}
} // namespace mlx::core
