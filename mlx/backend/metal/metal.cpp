// Copyright © 2023-2024 Apple Inc.

#include <cstdlib>
#include <future>
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
}

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

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

std::function<void()> make_task(
    array& arr,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p) {
  auto task = [arr, deps = std::move(deps), p = std::move(p)]() mutable {
    auto pool = new_scoped_memory_pool();
    for (auto& d : deps) {
      d.wait();
    }
    auto s = arr.primitive().stream();
    auto command_buffer = increment_command_buffer(s);
    auto outputs = arr.outputs();
    {
      // If the array is a tracer hold a reference
      // to its inputs so they don't get donated
      std::vector<array> inputs;
      if (arr.is_tracer()) {
        inputs = arr.inputs();
      }
      arr.primitive().eval_gpu(arr.inputs(), outputs);
    }
    std::vector<std::shared_ptr<array::Data>> buffers;
    for (auto& in : arr.inputs()) {
      buffers.push_back(in.data_shared_ptr());
    }
    for (auto& s : arr.siblings()) {
      buffers.push_back(s.data_shared_ptr());
    }
    if (!arr.is_tracer()) {
      arr.detach();
    }

    if (p) {
      metal::device(s.device).end_encoding(s.index);
      scheduler::notify_new_task(s);
      command_buffer->addCompletedHandler(
          [s, buffers = std::move(buffers), p = std::move(p)](
              MTL::CommandBuffer* cbuf) {
            p->set_value();
            scheduler::notify_task_completion(s);
            check_error(cbuf);
          });
      metal::device(s.device).commit_command_buffer(s.index);
    } else {
      command_buffer->addCompletedHandler(
          [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
            check_error(cbuf);
          });
    }
  };
  return task;
}

} // namespace mlx::core::metal
