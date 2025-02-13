// Copyright © 2023-2024 Apple Inc.
#include <memory>
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/event.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::metal {

bool is_available() {
  return true;
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
    auto& d = metal::device(s.device);
    auto command_buffer = d.get_command_buffer(s.index);
    d.increment_command_buffer_ops(s.index);

    for (auto& input : arr.inputs()) {
      if (input.event().valid() &&
          input.event().stream() != arr.primitive().stream()) {
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
      try {
        arr.primitive().eval_gpu(arr.inputs(), outputs);
      } catch (const std::exception& error) {
        abort_with_exception(error);
      }
    }

    // Use a set to de-dup input buffers
    std::unordered_set<std::shared_ptr<array::Data>> buffers;
    for (auto& in : arr.inputs()) {
      if (in.data_shared_ptr() != nullptr) {
        buffers.insert(in.data_shared_ptr());
      }
    }
    for (auto& s : arr.siblings()) {
      buffers.insert(s.data_shared_ptr());
    }
    if (!arr.is_tracer()) {
      arr.detach();
    }
    // Erase any input buffers that have a use count > 1
    // to keep them elligible for donation
    for (auto it = buffers.begin(); it != buffers.end();) {
      // Always hold buffers which could belong to outputs
      // TODO: this shouldn't be necessary, but metal validation
      // complains if buffers get released before the command buffer is
      // finished, even if the ready signal has fired
      if (!signal && it->use_count() > 1) {
        it = buffers.erase(it);
      } else {
        ++it;
      }
    }

    for (auto& out : outputs) {
      out.set_status(array::Status::evaluated);
    }

    if (signal ||
        d.get_command_buffer_ops(s.index) >= env::max_ops_per_buffer()) {
      if (signal) {
        encode_signal(arr.event());
      }
      d.end_encoding(s.index);
      scheduler::notify_new_task(s);
      command_buffer->addCompletedHandler(
          [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
            scheduler::notify_task_completion(s);
            check_error(cbuf);
          });
      d.commit_command_buffer(s.index);
      d.get_command_buffer(s.index);
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
    auto pool = new_scoped_memory_pool();
    auto& d = metal::device(s.device);
    auto cb = d.get_command_buffer(s.index);
    cb->retain();
    d.end_encoding(s.index);
    d.commit_command_buffer(s.index);
    cb->waitUntilCompleted();
    check_error(cb);
    cb->release();
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
