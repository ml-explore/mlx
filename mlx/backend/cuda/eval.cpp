// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core::gpu {

bool is_available() {
  return true;
}

void new_stream(Stream s) {
  // Force initalization of cuda, so cuda runtime get destroyed at last.
  cudaFree(nullptr);
  // Ensure the static stream objects get created.
  cu::get_command_encoder(s);
  // The main thread is safe to free buffers.
  cu::allocator().register_this_thread();
}

void eval(array& arr) {
  nvtx3::scoped_range r("gpu::eval");
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

  auto& encoder = cu::get_command_encoder(arr.primitive().stream());
  if (encoder.has_gpu_work()) {
    // Keep used buffers alive until kernel finishes running.
    std::unordered_set<std::shared_ptr<array::Data>> buffers;
    for (auto& in : arr.inputs()) {
      buffers.insert(in.data_shared_ptr());
    }
    for (auto& s : arr.siblings()) {
      buffers.insert(s.data_shared_ptr());
    }
    // Remove the output if it was donated to by an input.
    if (auto it = buffers.find(arr.data_shared_ptr()); it != buffers.end()) {
      buffers.erase(it);
    }
    encoder.add_completed_handler([buffers = std::move(buffers)]() {});
  }
  encoder.end_encoding();
}

void finalize(Stream s) {
  nvtx3::scoped_range r("gpu::finalize");
  cu::get_command_encoder(s).commit();
}

void synchronize(Stream s) {
  nvtx3::scoped_range r("gpu::synchronize");
  cu::get_command_encoder(s).synchronize();
}

} // namespace mlx::core::gpu
