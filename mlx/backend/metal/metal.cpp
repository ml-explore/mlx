// Copyright © 2023-2024 Apple Inc.
#include <cstdlib>
#include <memory>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
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

std::function<void()> make_task(array arr, bool signal) {
  auto task = [arr = std::move(arr), signal]() mutable {
    auto pool = new_scoped_memory_pool();
    auto s = arr.primitive().stream();
    auto command_buffer = increment_command_buffer(s);
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
      buffers.push_back(in.data_shared_ptr());
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

std::function<void()> make_synchronize_task(
    Stream s,
    std::shared_ptr<std::promise<void>> p) {
  return [s, p = std::move(p)]() {
    auto& d = metal::device(s.device);
    auto cb = d.get_command_buffer(s.index);
    if (cb == nullptr) {
      cb = d.new_command_buffer(s.index);
    } else {
      d.end_encoding(s.index);
    }
    d.commit_command_buffer(s.index);
    cb->waitUntilCompleted();
    check_error(cb);
    p->set_value();
  };
}

void start_capture(std::string path, id object) {
  auto pool = new_scoped_memory_pool();

  auto descriptor = MTL::CaptureDescriptor::alloc()->init();
  descriptor->setCaptureObject(object);

  if (!path.empty()) {
    auto string = NS::String::string(path.c_str(), NS::UTF8StringEncoding);
    auto url = NS::URL::fileURLWithPath(string);
    descriptor->setDestination(MTL::CaptureDestinationGPUTraceDocument);
    descriptor->setOutputURL(url);
  }

  auto manager = MTL::CaptureManager::sharedCaptureManager();
  NS::Error* error;
  bool started = manager->startCapture(descriptor, &error);
  descriptor->release();
  if (!started) {
    std::ostringstream msg;
    msg << "[metal::start_capture] Failed to start: "
        << error->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

void start_capture(std::string path) {
  auto& device = metal::device(mlx::core::Device::gpu);
  return start_capture(path, device.mtl_device());
}

void stop_capture() {
  auto pool = new_scoped_memory_pool();
  auto manager = MTL::CaptureManager::sharedCaptureManager();
  manager->stopCapture();
}

} // namespace mlx::core::metal
