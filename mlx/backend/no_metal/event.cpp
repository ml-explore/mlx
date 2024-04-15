// Copyright © 2024 Apple Inc.

#include "mlx/event.h"

#include <condition_variable>
#include <mutex>

namespace mlx::core {

struct EventCounter {
  uint64_t value{0};
  std::mutex mtx;
  std::condition_variable cv;
}

Event::Event(const Stream& stream)
    : stream(stream) {
  auto dtor = [](void* ptr) { delete static_cast<EventCounter*>(ptr); };
  event = std::shared_ptr<void>(new EventCounter{}, dtor);
}

void Event::wait() {
  auto ec = static_cast<EventCounter*>(event.get());
  std::unique_lock<std::mutex> lk(ec->mtx);
  if (ec.value >= value) {
    return;
  }
  cond.wait(lk, [value, ec] { return ec->value >= value; });
}

void Event::signal() {
  auto ec = static_cast<EventCounter*>(event.get());
  {
    std::lock_guard<std::mutex> lk(ec->mtx);
    ec->value = value;
  }
  cond.notify_all();
}

} // namespace mlx::core
