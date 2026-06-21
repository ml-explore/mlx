// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/cpu/encoder.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>

namespace mlx::core {

struct EventCounter {
  uint64_t value{0};
  std::exception_ptr error;
  std::mutex mtx;
  std::condition_variable cv;
};

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) { delete static_cast<EventCounter*>(ptr); };
  event_ = std::shared_ptr<void>(new EventCounter{}, dtor);
}

void Event::wait() {
  auto ec = static_cast<EventCounter*>(event_.get());
  std::unique_lock<std::mutex> lk(ec->mtx);
  ec->cv.wait(
      lk, [value = value(), ec] { return ec->value >= value || ec->error; });
  if (ec->error) {
    auto error = std::move(ec->error);
    ec->error = nullptr;
    std::rethrow_exception(error);
  }
}

void Event::wait(Stream stream) {
  cpu::get_command_encoder(stream).dispatch([*this]() mutable { wait(); });
}

void Event::signal(Stream stream) {
  cpu::get_command_encoder(stream).dispatch_unchecked([*this]() mutable {
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
  std::exception_ptr error;
  {
    std::lock_guard<std::mutex> lk(ec->mtx);
    if (ec->error) {
      error = std::move(ec->error);
      ec->error = nullptr;
    } else {
      return (ec->value >= value());
    }
  }
  std::rethrow_exception(error);
}

void Event::set_error(std::exception_ptr error) {
  auto ec = static_cast<EventCounter*>(event_.get());
  {
    std::lock_guard<std::mutex> lk(ec->mtx);
    if (!ec->error) {
      ec->error = std::move(error);
    }
    ec->value = std::max(ec->value, value());
  }
  ec->cv.notify_all();
}

std::exception_ptr Event::error() const {
  auto ec = static_cast<EventCounter*>(event_.get());
  std::lock_guard<std::mutex> lk(ec->mtx);
  return ec->error;
}
} // namespace mlx::core
