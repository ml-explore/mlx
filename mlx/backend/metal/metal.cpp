// Copyright © 2023 Apple Inc.

#include <cstdlib>
#include <future>
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::metal {

int max_ops_per_buffer() {
  auto get_val = []() {
    if (const char* buff_str = std::getenv("MLX_MAX_OPS_PER_BUFFER")) {
      return atoi(buff_str);
    } else {
      return 10;
    }
  };
  static int max_ops_per_buffer_ = get_val();
  return max_ops_per_buffer_;
}

#define MAX_OPS_PER_BUFFER max_ops_per_buffer()

MTL::CommandBuffer* increment_command_buffer(Stream s) {
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);
  if (command_buffer == nullptr ||
      d.get_command_buffer_ops(s.index) >= MAX_OPS_PER_BUFFER) {
    if (command_buffer != nullptr) {
      d.end_encoding(s.index);
      scheduler::notify_new_task(s);
      command_buffer->addCompletedHandler(
          [s](MTL::CommandBuffer*) { scheduler::notify_task_completion(s); });
      d.commit_command_buffer(s.index);
    }
    command_buffer = d.new_command_buffer(s.index);
  }
  d.increment_command_buffer_ops(s.index);
  return command_buffer;
}

std::function<void()> make_task(
    GraphNode g,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p) {
  auto task = [arr, deps = std::move(deps), p = std::move(p)]() mutable {
    auto pool = new_scoped_memory_pool();
    for (auto& d : deps) {
      d.wait();
    }
    auto s = g.primitive().stream();
    auto command_buffer = increment_command_buffer(s);
    g.primitive().eval_gpu(g.inputs(), g.outputs());
    if (p) {
      metal::device(s.device).end_encoding(s.index);
      scheduler::notify_new_task(s);
      command_buffer->addCompletedHandler(
          [s, arr, p = std::move(p)](MTL::CommandBuffer*) mutable {
            if (!arr.is_tracer()) {
              arr.detach();
            }
            p->set_value();
            scheduler::notify_task_completion(s);
          });
      metal::device(s.device).commit_command_buffer(s.index);
    } else {
      command_buffer->addCompletedHandler(
          [s, arr](MTL::CommandBuffer*) mutable {
            if (!arr.is_tracer()) {
              arr.detach();
            }
          });
    }
  };
  return task;
}

} // namespace mlx::core::metal
