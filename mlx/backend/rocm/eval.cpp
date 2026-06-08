// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>

namespace mlx::core::gpu {

void init() {
  hipFree(nullptr);
}

void new_stream(Stream s) {
  rocm::HipEvent(hipEventDefault);
  rocm::get_command_encoder(s);
}

void eval(array& arr) {
  auto outputs = arr.outputs();
  {
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  auto& encoder = rocm::get_command_encoder(arr.primitive().stream());
  for (auto& in : arr.inputs()) {
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
  rocm::get_command_encoder(s).commit();
}

void synchronize(Stream s) {
  rocm::get_command_encoder(s).synchronize();
}

void clear_streams() {
  rocm::clear_all_encoders();
}

} // namespace mlx::core::gpu

// --- GPU memcpy for direct KV cache writes ---
extern "C" void mlx_gpu_memcpy_async(void* dst, const void* src, size_t bytes) {
  auto& enc = mlx::core::rocm::get_command_encoder(
      mlx::core::default_stream(mlx::core::Device::gpu));
  enc.launch_kernel([=](hipStream_t stream) {
    (void)hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, stream);
  });
}

// --- Arena + Graph wrappers (called from engine code without HIP headers) ---
namespace mlx::core {

bool gpu_arena_begin(size_t capacity) {
  return rocm::allocator().arena().begin(capacity);
}
void gpu_arena_reset() {
  rocm::allocator().arena().reset();
}
void gpu_arena_end() {
  rocm::allocator().arena().end();
}
size_t gpu_arena_used() {
  return rocm::allocator().arena().used();
}
bool gpu_arena_active() {
  return rocm::allocator().arena().active();
}

static rocm::CommandEncoder& graph_encoder() {
  return rocm::get_command_encoder(default_stream(Device::gpu));
}

bool gpu_graph_begin_capture() {
  graph_encoder().begin_capture();
  return true;
}
bool gpu_graph_end_capture() {
  return graph_encoder().end_capture();
}
bool gpu_graph_replay() {
  return graph_encoder().replay();
}
void gpu_graph_reset() {
  graph_encoder().reset_graph();
}
bool gpu_graph_available() {
  return graph_encoder().has_graph();
}

} // namespace mlx::core
