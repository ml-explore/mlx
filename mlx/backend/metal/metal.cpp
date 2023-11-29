#include <cstdlib>
#include <future>
#include <memory>

#include "mlx/array.h"
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
    array& arr,
    std::vector<std::shared_future<void>> deps,
    std::shared_ptr<std::promise<void>> p,
    bool retain_graph) {
  auto task =
      [retain_graph, arr, deps = std::move(deps), p = std::move(p)]() mutable {
        for (auto& d : deps) {
          d.wait();
        }
        auto s = arr.primitive().stream();
        auto command_buffer = increment_command_buffer(s);
        arr.primitive().eval_gpu(arr.inputs(), arr);
        if (p) {
          metal::device(s.device).end_encoding(s.index);
          scheduler::notify_new_task(s);
          command_buffer->addCompletedHandler(
              [retain_graph, s, arr, p = std::move(p)](
                  MTL::CommandBuffer*) mutable {
                if (!retain_graph) {
                  arr.detach();
                }
                p->set_value();
                // Signal this thread to clear the pool on a synchroniztion.
                scheduler::enqueue(s, []() {
                  thread_autorelease_pool()->release();
                  thread_autorelease_pool() =
                      NS::AutoreleasePool::alloc()->init();
                });
                scheduler::notify_task_completion(s);
              });
          metal::device(s.device).commit_command_buffer(s.index);
        } else {
          command_buffer->addCompletedHandler(
              [retain_graph, s, arr](MTL::CommandBuffer*) mutable {
                if (!retain_graph) {
                  arr.detach();
                }
              });
        }
      };
  return task;
}

} // namespace mlx::core::metal
