// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/gpu/available.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core::gpu {

// Can be tuned with MLX_MAX_OPS_PER_BUFFER
constexpr int default_max_nodes_per_graph = 20;

bool is_available() {
  return true;
}

void new_stream(Stream s) {
  // Force initalization of CUDA, so CUDA runtime get destroyed at last.
  cudaFree(nullptr);
  // Make sure CUDA event pool get destroyed after device and stream.
  cu::CudaEvent::init_pool();
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

  auto& stream = arr.primitive().stream();
  auto& encoder = cu::get_command_encoder(stream);
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

  if (encoder.get_num_ops() >=
      env::max_ops_per_buffer(default_max_nodes_per_graph)) {
    scheduler::notify_new_task(stream);
    encoder.add_completed_handler(
        [stream]() { scheduler::notify_task_completion(stream); });
    encoder.commit();
  }
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
