// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/cublas_utils.h"
#include "mlx/backend/cuda/cudnn_utils.h"
#include "mlx/backend/cuda/event.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core::gpu {

void init() {
  // Force initalization of CUDA, so CUDA runtime get destroyed last.
  cudaFree(nullptr);
  // Make sure CUDA event pool get destroyed after device and stream.
  mlx::core::cu::CudaEvent::init_pool();
}

void new_stream(Stream s) {
  // Make sure the handles get destroyed after CommandEncoder.
  init_cublas_handles_cache();
  init_cudnn_handles_cache();
  init_cudnn_conv_cache();
  init_cudnn_sdpa_cache();
  // Create CommandEncoder.
  assert(s.device == Device::gpu);
  auto& encoders = cu::get_command_encoders();
  auto& d = cu::device(s.device);
  encoders.try_emplace(s.index, d);
}

void eval(array& arr) {
  nvtx3::scoped_range r("gpu::eval");
  // Ensure CUDA context is active on this thread. Required when MLX is called
  // from threads that have not yet established a CUDA context (e.g. thread
  // pools, language runtimes that migrate work across OS threads).
  cu::device(arr.primitive().stream().device).make_current();
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

  if (encoder.needs_commit()) {
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
