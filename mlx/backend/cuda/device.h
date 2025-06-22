// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/stream.h"

#include <cublasLt.h>
#include <thrust/execution_policy.h>

#include <unordered_map>

namespace mlx::core::cu {

class CommandEncoder {
 public:
  struct CaptureContext {
    CaptureContext(CommandEncoder& enc);
    ~CaptureContext();
    cudaGraph_t graph;
    CommandEncoder& enc;
  };

  explicit CommandEncoder(Device& d);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  CaptureContext capture_context() {
    return CaptureContext{*this};
  }
  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  void add_completed_handler(std::function<void()> task);
  void maybe_commit();
  void commit();

  CudaStream& stream() {
    return stream_;
  }

  // Wait until kernels and completion handlers are finished
  void synchronize();


 private:
  CudaStream stream_;
  cudaGraph_t graph_;
  Worker worker_;
  int num_ops_{0};
  std::vector<std::shared_ptr<array::Data>> temporaries_;
  std::vector<std::uintptr_t> active_inputs_;
  std::vector<std::uintptr_t> active_outputs_;
  std::unordered_map<std::uintptr_t, cudaGraphNode_t> node_map_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current cuda device, required by some cuda calls.
  void make_current();

  CommandEncoder& get_command_encoder(Stream s);

  int cuda_device() const {
    return device_;
  }
  int compute_capability_major() const {
    return compute_capability_major_;
  }
  int compute_capability_minor() const {
    return compute_capability_minor_;
  }
  cublasLtHandle_t lt_handle() const {
    return lt_;
  }

 private:
  int device_;
  int compute_capability_major_;
  int compute_capability_minor_;
  cublasLtHandle_t lt_;
  std::unordered_map<int, CommandEncoder> encoders_;
};

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);

// Return an execution policy that does not sync for result.
// Note that not all thrust APIs support async policy, confirm before using.
inline auto thrust_policy(cudaStream_t stream) {
  // TODO: Connect thrust's custom allocator with mlx's allocator.
  return thrust::cuda::par_nosync.on(stream);
}

} // namespace mlx::core::cu
