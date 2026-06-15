// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/stream.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

// Only include thrust headers when compiling with HIP compiler
// (thrust headers have dependencies on CUDA/HIP-specific headers)
#ifdef __HIPCC__
#include <thrust/execution_policy.h>
#endif

#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mlx::core::rocm {

// Forward declaration
class Device;
class Worker;

class CommandEncoder {
 public:
  explicit CommandEncoder(Device& d);
  ~CommandEncoder();

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  template <typename F>
  void launch_kernel(F&& func);

  void add_temporary(const array& arr);

  void add_completed_handler(std::function<void()> task);
  void maybe_commit();
  void commit();

  Device& device() {
    return device_;
  }

  HipStream& stream() {
    return stream_;
  }

  // Wait until kernels and completion handlers are finished
  void synchronize();

  // --- Graph capture API ---
  // Begin recording all kernel launches into a HIP graph.
  // While capturing, launch_kernel dispatches are recorded (not executed).
  void begin_capture();

  // End recording and instantiate the captured graph.
  // Returns true if capture succeeded (graph is ready to replay).
  bool end_capture();

  // Replay the previously captured graph. All recorded kernels execute
  // in a single GPU dispatch. Returns false if no graph is available.
  // If sync is true (default) the call blocks until the replayed work
  // finishes. If false it only launches the graph onto the stream and
  // returns immediately — the caller must order any reads of the graph's
  // outputs after it on the SAME stream (subsequent MLX eval on the
  // generation stream does exactly this), which lets per-token sampling
  // pipeline instead of draining the GPU every token.
  bool replay(bool sync = true);

  // Returns true if a captured graph is ready to replay.
  bool has_graph() const {
    return graph_exec_ != nullptr;
  }

  // True while this encoder's stream is recording into a HIP graph. Used by the
  // Event layer to avoid recording completion events onto the captured stream
  // (they would be baked into the graph and never fire, deadlocking eval).
  bool capturing() const {
    return capturing_;
  }

  // Discard the captured graph.
  void reset_graph();

 private:
  Device& device_;
  HipStream stream_;
  std::unique_ptr<Worker> worker_;
  int node_count_{0};
  std::vector<std::shared_ptr<array::Data>> temporaries_;
  std::unordered_set<const array::Data*> temporary_ptrs_;
  bool capturing_{false};
  // Buffers allocated during capture are held alive here (not freed) so their
  // addresses stay valid and unique for the lifetime of the captured graph —
  // freeing them mid-capture would let later allocations reuse the same
  // address, aliasing distinct graph nodes. Released in reset_graph().
  std::vector<std::shared_ptr<array::Data>> capture_held_;
  hipGraph_t graph_{nullptr};
  hipGraphExec_t graph_exec_{nullptr};
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current HIP device, required by some HIP calls.
  void make_current();

  CommandEncoder& get_command_encoder(Stream s);
  void clear_encoders();

  int hip_device() const {
    return device_;
  }

  rocblas_handle get_rocblas_handle();
  void set_rocblas_stream(hipStream_t stream);

  // Check if rocBLAS is available for the current GPU architecture
  bool is_rocblas_available();

  // Check if rocBLAS bf16 GEMM works on this device (probed at init)
  bool is_rocblas_bf16_available();

  // True iff this device's gcnArchName is on the rocWMMA arch allowlist
  // (CDNA1/2/3 + RDNA3 dGPU + gfx1151 + RDNA4). Lazy-cached on first call.
  bool has_native_wmma();

 private:
  int device_;
  rocblas_handle rocblas_{nullptr};
  hipStream_t rocblas_stream_{nullptr};
  bool rocblas_initialized_{false};
  bool rocblas_available_{true};
  bool rocblas_bf16_probed_{false};
  bool rocblas_bf16_available_{false};
  bool wmma_probed_{false};
  bool has_native_wmma_{false};
  std::unordered_map<int, std::unique_ptr<CommandEncoder>> encoders_;
};

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);
void clear_all_encoders();

// True while a HIP graph capture is in progress on any stream. Lazy library
// inits that abort under capture (e.g. hipblasLtCreate) check this.
bool stream_capturing();

// Return an execution policy that does not sync for result.
// Only available when compiling with HIP compiler
#ifdef __HIPCC__
inline auto thrust_policy(hipStream_t stream) {
  return thrust::hip::par.on(stream);
}
#endif

// Template implementation (must be after Device is defined)
template <typename F>
void CommandEncoder::launch_kernel(F&& func) {
  device_.make_current();
  // When capturing, kernel launches are recorded into the HIP graph
  // automatically via hipStreamBeginCapture. No special handling needed —
  // hipLaunchKernel on a capturing stream records instead of executing.
  func(static_cast<hipStream_t>(stream_));
  node_count_++;
}

} // namespace mlx::core::rocm
