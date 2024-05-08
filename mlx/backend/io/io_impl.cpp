// Copyright © 2024 Apple Inc.

#include "mlx/backend/io/io_impl.h"
#include "mlx/backend/io/thread_pool.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::io {

namespace {

detail::ThreadPool& thread_pool() {
  static std::unique_ptr<detail::ThreadPool> pool_ptr;

  if (pool_ptr == nullptr) {
    pool_ptr = std::make_unique<detail::ThreadPool>(4);
  }

  return *pool_ptr;
}

} // namespace

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

    // Schedule the computation
    auto inputs = arr.inputs();
    auto outputs = arr.outputs();
    thread_pool().enqueue(
        [arr = std::move(arr), inputs, outputs, signal, stream]() mutable {
          // Perform the computation
          arr.primitive().eval_io(inputs, outputs);

          if (!arr.is_tracer()) {
            arr.detach();
          }

          if (signal) {
            thread_pool().barrier(
                [arr = std::move(arr)]() { arr.event().signal(); });
          }

          // Task computation done.
          scheduler::notify_task_completion(stream);
        },
        inputs,
        outputs);
  };
}

std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p) {
  return [p = std::move(p)]() {
    thread_pool().barrier().wait();
    p->set_value();
  };
}

} // namespace mlx::core::io
