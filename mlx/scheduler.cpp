// Copyright © 2023 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/backend/gpu/eval.h"

namespace mlx::core {

Stream default_stream(Device d) {
  if (!gpu::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[default_stream] Cannot get gpu stream without gpu backend.");
  }
  return scheduler::scheduler().get_default_stream(d);
}

void set_default_stream(Stream s) {
  if (!gpu::is_available() && s.device == Device::gpu) {
    throw std::invalid_argument(
        "[set_default_stream] Cannot set gpu stream without gpu backend.");
  }
  return scheduler::scheduler().set_default_stream(s);
}

Stream get_stream(int index) {
  return scheduler::scheduler().get_stream(index);
}

Stream new_stream(Device d) {
  if (!gpu::is_available() && d == Device::gpu) {
    throw std::invalid_argument(
        "[new_stream] Cannot make gpu stream without gpu backend.");
  }
  return scheduler::scheduler().new_stream(d);
}

Stream new_stream() {
  return scheduler::scheduler().new_stream(default_device());
}

void synchronize(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    auto p = std::make_shared<std::promise<void>>();
    std::future<void> f = p->get_future();
    scheduler::enqueue(s, [p = std::move(p)]() { p->set_value(); });
    f.wait();
  } else {
    gpu::synchronize(s);
  }
}

void synchronize() {
  synchronize(default_stream(default_device()));
}

namespace scheduler {

/** A singleton scheduler to manage devices, streams, and task execution. */
Scheduler& scheduler() {
  // Leak the scheduler on Windows to avoid joining threads on exit, can be
  // removed after Visual Studio fixes bug:
  // https://developercommunity.visualstudio.com/t/1654756
#ifdef _WIN32
  static Scheduler* scheduler = new Scheduler;
  return *scheduler;
#else
  static Scheduler scheduler;
  return scheduler;
#endif
}

} // namespace scheduler
} // namespace mlx::core
