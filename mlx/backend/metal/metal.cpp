// Copyright © 2023-2024 Apple Inc.
#include <cstdlib>
#include <future>
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
}

int get_env_val(const char* name, int fallback) {
  if (const char* buff_str = std::getenv(name)) {
    return atoi(buff_str);
  } else {
    return fallback;
  }
}

constexpr int default_max_ops[3] = {
    10, // Category::small
    30, // Category::medium
    50, // Category::large
};

constexpr int default_max_mb[3] = {
    20, // Category::small
    50, // Category::medium
    50, // Category::large
};

int max_ops_per_buffer(metal::Device::Category category) {
  static int max_ops_per_buffer_ =
      get_env_val("MLX_MAX_OPS_PER_BUFFER", default_max_ops[category]);
  return max_ops_per_buffer_;
}

int max_mb_per_buffer(metal::Device::Category category) {
  static int max_mb_per_buffer_ =
      get_env_val("MLX_MAX_MB_PER_BUFFER", default_max_mb[category]);
  return max_mb_per_buffer_;
}

// Get an active command buffer. This function will possibly make a new
// command buffer if the commit conditions are satisfied.
MTL::CommandBuffer* fetch_command_buffer(Stream s) {
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

  auto check_commit = [&d, &s]() {
    auto& cb_info = d.get_command_buffer_info(s.index);
    return (
        cb_info.ops >= max_ops_per_buffer(d.category()) ||
        (cb_info.bytes >> 20) >= max_mb_per_buffer(d.category()));
  };
  if (command_buffer == nullptr || check_commit()) {
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

std::function<void()> make_task(array arr, bool signal) {
  auto task = [arr = std::move(arr), signal]() mutable {
    auto pool = new_scoped_memory_pool();
    auto s = arr.primitive().stream();

    auto command_buffer = fetch_command_buffer(s);
    for (auto& input : arr.inputs()) {
      if (input.event().valid() &&
          input.event().stream() != arr.primitive().stream()) {
        // TODO, consider committing the buffer and encoding a wait in the new
        // buffer rather than on the task thread
        input.event().wait();
      }
    }

    auto outputs = arr.outputs();
    {
      // If the array is a tracer hold a reference
      // to its inputs so they don't get donated
      std::vector<array> inputs;
      if (arr.is_tracer()) {
        inputs = arr.inputs();
      }

      debug_set_primitive_buffer_label(command_buffer, arr.primitive());
      arr.primitive().eval_gpu(arr.inputs(), outputs);
    }
    std::vector<std::shared_ptr<array::Data>> buffers;
    for (auto& in : arr.inputs()) {
      if (in.data_shared_ptr() != nullptr) {
        buffers.push_back(in.data_shared_ptr());
      }
    }
    for (auto& s : arr.siblings()) {
      buffers.push_back(s.data_shared_ptr());
    }
    if (!arr.is_tracer()) {
      arr.detach();
    }

    if (signal) {
      metal::device(s.device).end_encoding(s.index);
      command_buffer->encodeSignalEvent(
          static_cast<MTL::Event*>(arr.event().raw_event().get()),
          arr.event().value());
      scheduler::notify_new_task(s);
      command_buffer->addCompletedHandler(
          [s, buffers = std::move(buffers), event = arr.event()](
              MTL::CommandBuffer* cbuf) {
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

bool start_capture(std::string path, id object) {
  auto pool = new_scoped_memory_pool();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init();
  descriptor->setCaptureObject(object);

  if (path.length() > 0) {
    auto string = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }

  auto manager = MTL::CaptureManager::sharedCaptureManager();
  return manager->startCapture(descriptor, nullptr);
}

bool start_capture(std::string path) {
  auto& device = metal::device(mlx::core::Device::gpu);
  return start_capture(path, device.mtl_device());
}

void stop_capture() {
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
}

} // namespace mlx::core::metal
