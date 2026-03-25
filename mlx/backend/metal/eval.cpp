// Copyright © 2023-2024 Apple Inc.
#include <memory>
#include <mutex>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

namespace {
// Thread-safe deferred error from Metal completion handlers.
// Completion handlers run on GCD threads where C++ exceptions
// hit std::terminate. Instead, we store the error and re-throw
// at the next eval() or synchronize() call.
std::mutex deferred_error_mutex;
std::string deferred_error_message;

void set_deferred_error(const std::string& msg) {
  std::lock_guard<std::mutex> lock(deferred_error_mutex);
  if (deferred_error_message.empty()) {
    deferred_error_message = msg;
  }
}

void check_deferred_error() {
  std::lock_guard<std::mutex> lock(deferred_error_mutex);
  if (!deferred_error_message.empty()) {
    std::string msg = std::move(deferred_error_message);
    deferred_error_message.clear();
    throw std::runtime_error(msg);
  }
}
} // namespace

void init() {}

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).get_command_encoder(stream.index);
  }
}

inline void check_error(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    throw std::runtime_error(msg.str());
  }
}

// Safe version for Metal completion handlers (GCD callbacks).
// Cannot throw — stores error for deferred propagation.
inline void check_error_deferred(MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    set_deferred_error(msg.str());
  }
}

void eval(array& arr) {
  // Re-throw any deferred error from a prior completion handler
  check_deferred_error();
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);
  auto command_buffer = d.get_command_buffer(s.index);

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
  std::unordered_set<std::shared_ptr<array::Data>> buffers;
  for (auto& in : arr.inputs()) {
    buffers.insert(in.data_shared_ptr());
  }
  for (auto& s : arr.siblings()) {
    buffers.insert(s.data_shared_ptr());
  }
  // Remove the output if it was donated to by an input
  if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
    buffers.erase(it);
  }

  if (d.command_buffer_needs_commit(s.index)) {
    d.end_encoding(s.index);
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          check_error_deferred(cbuf);
        });
    d.commit_command_buffer(s.index);
  } else {
    command_buffer->addCompletedHandler(
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          check_error_deferred(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  d.end_encoding(s.index);
  cb->addCompletedHandler(
      [](MTL::CommandBuffer* cbuf) { check_error_deferred(cbuf); });
  d.commit_command_buffer(s.index);
}

void synchronize(Stream s) {
  check_deferred_error();
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto cb = d.get_command_buffer(s.index);
  cb->retain();
  d.end_encoding(s.index);
  d.commit_command_buffer(s.index);
  cb->waitUntilCompleted();
  check_error(cb);
  cb->release();
}

} // namespace mlx::core::gpu
