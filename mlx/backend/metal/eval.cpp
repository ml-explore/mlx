// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void new_stream(Stream stream) {
  if (stream.device == mlx::core::Device::gpu) {
    metal::device(stream.device).new_queue(stream.index);
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

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  auto& d = metal::device(s.device);

  auto outputs = arr.outputs();
  // Host-side prep stays outside the stream op lock.
  std::vector<array> tracer_inputs;
  if (arr.is_tracer()) {
    tracer_inputs = arr.inputs();
  }

  auto lk = d.lock_stream_ops(s.index);
  auto command_buffer = d.get_command_buffer(s.index);
  debug_set_primitive_buffer_label(command_buffer, arr.primitive());
  // Degraded split: primitive eval currently performs both plan+encode.
  // Keep full primitive work under op lock until per-primitive prepare/encode
  // split is implemented.
  arr.primitive().eval_gpu(arr.inputs(), outputs);
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
    d.end_encoding(s.index, lk);
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
        [buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          check_error(cbuf);
        });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  auto lk = d.lock_stream_ops(s.index);
  auto cb = d.get_command_buffer(s.index);
  d.end_encoding(s.index, lk);
  cb->addCompletedHandler([](MTL::CommandBuffer* cbuf) { check_error(cbuf); });
  d.commit_command_buffer(s.index);
  d.get_command_buffer(s.index);
}

void synchronize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& d = metal::device(s.device);
  MTL::CommandBuffer* cb = nullptr;
  {
    auto lk = d.lock_stream_ops(s.index);
    cb = d.get_command_buffer(s.index);
    cb->retain();
    d.end_encoding(s.index, lk);
    d.commit_command_buffer(s.index);
  }
  cb->waitUntilCompleted();
  check_error(cb);
  cb->release();
}

} // namespace mlx::core::gpu
