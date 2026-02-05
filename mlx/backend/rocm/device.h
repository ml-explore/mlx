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

  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

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

 private:
  Device& device_;
  HipStream stream_;
  std::unique_ptr<Worker> worker_;
  int node_count_{0};
  std::vector<std::shared_ptr<array::Data>> temporaries_;
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

  int hip_device() const {
    return device_;
  }

  rocblas_handle get_rocblas_handle();
  
  // Check if rocBLAS is available for the current GPU architecture
  bool is_rocblas_available();

 private:
  int device_;
  rocblas_handle rocblas_{nullptr};
  bool rocblas_initialized_{false};
  bool rocblas_available_{true};
  std::unordered_map<int, std::unique_ptr<CommandEncoder>> encoders_;
};

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);

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
  func(static_cast<hipStream_t>(stream_));
}

} // namespace mlx::core::rocm
