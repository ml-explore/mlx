// Copyright © 2025 Apple Inc.

#include "mlx/backend/gpu/eval.h"
#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace mlx::core::gpu {

void init() {
  // Initialize the SELECTED default GPU's primary context — not device 0. On a
  // multi-GPU host, creating a context/queue on the other GPU (the APU) too is
  // what differs from HIP_VISIBLE_DEVICES, and that cross-device queue coexistence
  // is what wedges the discrete GPU's command queue over a TB5 link. Touch only
  // the chosen device so the runtime behaves as if it were the only one.
  auto d = mlx::core::default_device();
  if (d.type == mlx::core::Device::gpu) {
    (void)hipSetDevice(d.index);
  }
  hipFree(nullptr);
}

void new_stream(Stream s) {
  // Bind the stream's device FIRST (creates/selects its Device + context), then
  // warm the event pool on that device — creating the HipEvent before binding
  // would put it (and its queue interaction) on whatever device is current.
  rocm::get_command_encoder(s);
  rocm::HipEvent(hipEventDefault);
}

// Ops whose kernels corrupt when batched into a multi-node HIP graph with
// neighbors (a ROCm CLR kernarg-pool interaction; found by per-op force-execute
// bisection). Isolate them: flush the graph before AND after so they run alone.
static bool is_graph_split_op(const char* name) {
  return std::strcmp(name, "Concatenate") == 0;
}

void eval(array& arr) {
  auto outputs = arr.outputs();
  auto& encoder = rocm::get_command_encoder(arr.primitive().stream());
  const bool split =
      rocm::use_hip_graphs() && is_graph_split_op(arr.primitive().name());
  if (split) {
    encoder.commit(); // flush ops accumulated before this one
  }
  // Bind the stream's device before eval_gpu so output buffers allocate on the
  // same device the kernels run on. Otherwise (multi-GPU) outputs land on
  // whatever device is current (often device 0) while kernels run on the
  // stream's device, stranding the model on the wrong GPU.
  encoder.device().make_current();
  {
    std::vector<array> inputs;
    if (arr.is_tracer()) {
      inputs = arr.inputs();
    }
    arr.primitive().eval_gpu(arr.inputs(), outputs);
  }

  for (auto& in : arr.inputs()) {
    if (in.data_shared_ptr() != arr.data_shared_ptr()) {
      encoder.add_temporary(in);
    }
  }
  for (auto& s : arr.siblings()) {
    encoder.add_temporary(s);
  }

  if (rocm::use_hip_graphs()) {
    auto& stream = arr.primitive().stream();
    if (split || encoder.needs_commit()) {
      scheduler::notify_new_task(stream);
      encoder.add_completed_handler(
          [stream]() { scheduler::notify_task_completion(stream); });
      encoder.commit();
    }
  } else {
    encoder.maybe_commit();
  }

  // Bisection: batch ops [0, FORCE_FROM) into graphs (per cap), force-execute
  // (commit+sync = correct) every op >= FORCE_FROM. The smallest FORCE_FROM that
  // turns the output to garbage pinpoints the first op whose batching breaks.
  if (rocm::use_hip_graphs()) {
    static const int force_from = std::getenv("MLX_GRAPH_FORCE_FROM")
        ? std::atoi(std::getenv("MLX_GRAPH_FORCE_FROM"))
        : -1;
    if (force_from >= 0) {
      static int gidx = 0;
      int my = gidx++;
      if (my >= force_from) {
        encoder.commit();
        encoder.synchronize();
      }
      static const bool ftr = std::getenv("MLX_GRAPH_FORCE_TRACE") != nullptr;
      if (ftr && my >= force_from - 6 && my <= force_from + 1)
        fprintf(stderr, "[ff] op %d : %s\n", my, arr.primitive().name());
    }
  }
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
  // Use the SELECTED default device's stream, not Device::gpu (which is the
  // device TYPE = gpu index 0). On a multi-GPU box, --device 1 would otherwise
  // memcpy KV data on device 0's stream while the data lives on device 1.
  auto& enc = mlx::core::rocm::get_command_encoder(
      mlx::core::default_stream(mlx::core::default_device()));
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
size_t gpu_arena_desc_used() {
  return rocm::allocator().arena().desc_used();
}
void gpu_arena_reset_to(size_t byte_mark, size_t desc_mark) {
  rocm::allocator().arena().reset_to(byte_mark, desc_mark);
}
void gpu_arena_set_paused(bool p) {
  rocm::allocator().arena().set_paused(p);
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
  return rocm::get_command_encoder(default_stream(default_device()));
}

bool gpu_graph_begin_capture() {
  graph_encoder().begin_capture();
  return true;
}
bool gpu_graph_end_capture() {
  return graph_encoder().end_capture();
}
bool gpu_graph_replay() {
  return graph_encoder().replay(/*sync=*/true);
}
bool gpu_graph_replay_async() {
  return graph_encoder().replay(/*sync=*/false);
}
void gpu_graph_reset() {
  graph_encoder().reset_graph();
}
bool gpu_graph_available() {
  return graph_encoder().has_graph();
}

} // namespace mlx::core
