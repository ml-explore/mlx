// Copyright © 2023 Apple Inc.

#include "mlx/scheduler.h"
#include "mlx/backend/gpu/eval.h"

namespace mlx::core {

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

Scheduler::Scheduler() {
  gpu::init();
}

Scheduler::~Scheduler() = default;

void Scheduler::new_thread(Device::DeviceType type) {
  if (type == Device::gpu) {
    threads_.push_back(nullptr);
  } else {
    threads_.push_back(std::make_unique<StreamThread>());
  }
}

/** A singleton scheduler to manage devices, streams, and task execution. */
Scheduler& scheduler() {
  // Intentionally leaked to avoid the "static destruction order fiasco":
  // background threads (e.g. command buffer completion handlers) may
  // reference this singleton after other static objects are destroyed
  // during process teardown.
  static Scheduler* scheduler = new Scheduler;
  return *scheduler;
}

} // namespace scheduler
} // namespace mlx::core
