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
  // Force initalization of CUDA by creating an event, so the CUDA runtime and
  // our CUDA event pool get destroyed last.
  cu::CudaEvent(cudaEventDefault);
  // Ensure the static stream objects get created.
  cu::get_command_encoder(s);
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
  // Keep used buffers alive until kernel finishes running.
  for (auto& in : arr.inputs()) {
    // Except for the donated one.
    if (in.data_shared_ptr() != arr.data_shared_ptr()) {
      encoder.add_temporary(in);
    }
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }
  encoder.maybe_commit();
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
