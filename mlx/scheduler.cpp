// Copyright © 2023 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/metal/metal.h"

namespace mlx::core {

Stream default_stream(Device d) {
  if (!metal::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[default_stream] Cannot get gpu stream without gpu backend.");
  }
  return scheduler::scheduler().get_default_stream(d);
}

void set_default_stream(Stream s) {
  if (!metal::is_available() && s.device == Device::gpu) {
    throw std::invalid_argument(
        "[set_default_stream] Cannot set gpu stream without gpu backend.");
  }
  return scheduler::scheduler().set_default_stream(s);
}

Stream new_stream(Device d, int threads /* = 1 */) {
  if (!metal::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[new_stream] Cannot make gpu stream without gpu backend.");
  }
  if (d == Device::gpu && threads > 1) {
    throw std::invalid_argument(
        "[new_stream] Cannot make multi-threaded gpu stream.");
  }
  return scheduler::scheduler().new_stream(d, threads);
}

void synchronize(Stream s) {
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [p = std::move(p)]() { p->set_value(); });
  } else {
    scheduler::enqueue(s, metal::make_synchronize_task(s, std::move(p)));
  }
  f.wait();
}

void synchronize() {
  synchronize(default_stream(default_device()));
}

namespace scheduler {

/** A singleton scheduler to manage devices, streams, and task execution. */
Scheduler& scheduler() {
  static Scheduler scheduler;
  return scheduler;
}

} // namespace scheduler
} // namespace mlx::core
