// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/stream.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <thrust/execution_policy.h>

#include <unordered_map>

namespace mlx::core::rocm {

class CommandEncoder {
 public:
  explicit CommandEncoder(Device& d);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  void set_input_array(const array& arr);
  void set_output_array(const array& arr);

  template <typename F>
  void launch_kernel(F&& func) {
    device_.make_current();
    func(stream_);
  }

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
  Worker worker_;
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
  
  rocblas_handle rocblas_handle() const {
    return rocblas_;
  }

 private:
  int device_;
  rocblas_handle rocblas_;
  std::unordered_map<int, CommandEncoder> encoders_;
};

Device& device(mlx::core::Device device);
CommandEncoder& get_command_encoder(Stream s);

// Return an execution policy that does not sync for result.
inline auto thrust_policy(hipStream_t stream) {
  return thrust::hip::par.on(stream);
}

} // namespace mlx::core::rocm
