// Copyright © 2023 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/common/cpu_impl.h"
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

Stream new_stream(Device d) {
  if (!metal::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[new_stream] Cannot make gpu stream without gpu backend.");
  }
  return scheduler::scheduler().new_stream(d);
}

Stream new_stream() {
  return scheduler::scheduler().new_stream(default_device());
}

void synchronize(Stream s) {
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  switch (s.device.type) {
    case mlx::core::Device::cpu:
      scheduler::enqueue(s, cpu::make_synchronize_task(s, std::move(p)));
      break;
    case mlx::core::Device::gpu:
      scheduler::enqueue(s, metal::make_synchronize_task(s, std::move(p)));
      break;
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
