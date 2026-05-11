// Copyright © 2023-2024 Apple Inc.
#include <memory>

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

// Capture (don't throw) a Metal command-buffer error from an async
// completion handler. The handler runs on a Metal-managed dispatch
// thread; throwing through Objective-C frames hits _objc_terminate ->
// abort() and crashes the whole process. Storing the message and
// re-throwing on the next user-thread eval()/finalize() call keeps
// the error catchable by Python / C++ callers.
inline void capture_async_error(const Stream& s, MTL::CommandBuffer* cbuf) {
  if (cbuf->status() == MTL::CommandBufferStatusError) {
    std::ostringstream msg;
    msg << "[METAL] Command buffer execution failed: "
        << cbuf->error()->localizedDescription()->utf8String();
    scheduler::capture_error(s, msg.str());
  }
}

// Re-throw any previously-captured async Metal error on the calling
// thread. Called at the entry of every user-facing eval()/finalize()
// so failures from the prior step surface before more work is queued.
inline void throw_if_captured(const Stream& s) {
  auto msg = scheduler::take_error(s);
  if (!msg.empty()) {
    throw std::runtime_error(msg);
  }
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
  // Surface any async Metal error captured by a prior step's completion
  // handler before queuing more work on this stream.
  throw_if_captured(s);
  auto& encoder = metal::get_command_encoder(s);
  auto* command_buffer = encoder.get_command_buffer();

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

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          scheduler::notify_task_completion(s);
          capture_async_error(s, cbuf);
        });
    encoder.commit();
  } else {
    command_buffer->addCompletedHandler(
        [s, buffers = std::move(buffers)](MTL::CommandBuffer* cbuf) {
          capture_async_error(s, cbuf);
        });
  }
}

void finalize(Stream s) {
  // Surface any prior async Metal error before sealing the stream.
  throw_if_captured(s);
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  cb->addCompletedHandler(
      [s](MTL::CommandBuffer* cbuf) { capture_async_error(s, cbuf); });
  encoder.commit();
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize();
  // CommandEncoder::synchronize only checks the FINAL command buffer.
  // An async failure on an earlier (already-completed) command buffer
  // landed in the captured-error slot — surface it here too so
  // synchronize() is a true "all my queued work succeeded" guarantee.
  throw_if_captured(s);
}

void clear_streams() {
  metal::get_command_encoders().clear();
  // After tearing down encoder state, also clear any poisoned-stream
  // flags so the streams are usable again. (The flags would otherwise
  // outlive the encoders they refer to and refuse all subsequent
  // work — see scheduler::StreamThread.)
  scheduler::reset_all_errors();
}

} // namespace mlx::core::gpu
