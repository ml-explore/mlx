// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/cpu_impl.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::cpu {

std::function<void()> make_task(array arr, bool signal) {
  return [arr = std::move(arr), signal]() mutable {
    auto stream = arr.primitive().stream();

    // Wait on inputs coming from different streams/devices.
    for (auto& input : arr.inputs()) {
      if (input.event().valid() && input.event().stream() != stream) {
        input.event().wait();
      }
    }

    // Task computation actually starting.
    scheduler::notify_new_task(stream);

    // Perform the computation
    auto outputs = arr.outputs();
    arr.primitive().eval_cpu(arr.inputs(), outputs);

    // Check if we need to detach and signal other arrays waiting for the
    // result to be ready.
    if (!arr.is_tracer()) {
      arr.detach();
    }
    if (signal) {
      arr.event().signal();
    }

    // Task computation done.
    scheduler::notify_task_completion(stream);
  };
}

std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p) {
  return [p = std::move(p)]() { p->set_value(); };
}

} // namespace mlx::core::cpu
