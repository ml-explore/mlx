// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"
#include "mlx/utils.h"

namespace mlx::core::gpu {

void init() {}

void new_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

void new_thread_unsafe_stream(Stream s) {
  assert(s.device == Device::gpu);
  auto& encoders = metal::get_global_command_encoders();
  auto& d = metal::device(s.device);
  encoders.try_emplace(s.index, d, s.index, d.residency_set());
}

void eval(array& arr) {
  auto pool = metal::new_scoped_memory_pool();
  auto s = arr.primitive().stream();
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
  // Skip a donated output's buffer since holding it blocks allocator reuse.
  const auto& out_data = arr.data_shared_ptr();
  for (auto& in : arr.inputs()) {
    if (in.data_shared_ptr() != out_data) {
      encoder.hold_buffer(in.data_shared_ptr());
    }
  }
  for (auto& sib : arr.siblings()) {
    if (sib.data_shared_ptr() != out_data) {
      encoder.hold_buffer(sib.data_shared_ptr());
    }
  }

  if (encoder.needs_commit()) {
    encoder.end_encoding();
    scheduler::notify_new_task(s);
    encoder.commit([s]() { scheduler::notify_task_completion(s); });
  }
}

void finalize(Stream s) {
  auto pool = metal::new_scoped_memory_pool();
  auto& encoder = metal::get_command_encoder(s);
  auto* cb = encoder.get_command_buffer();
  encoder.end_encoding();
  encoder.commit();
}

void synchronize(Stream s) {
  metal::get_command_encoder(s).synchronize();
}

void clear_streams() {
  metal::get_command_encoders().clear();
  if (is_main_thread()) {
    metal::get_global_command_encoders().clear();
  }
}

} // namespace mlx::core::gpu
