// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/stream.h"

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>

#include <unordered_map>

namespace mlx::core {

namespace rocm {

class Device;
class CommandEncoder;

class DeviceStream {
 public:
  explicit DeviceStream(Device& device);

  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  // Wait until kernels in the stream complete.
  void synchronize();

  // Return a HIP stream for launching kernels.
  hipStream_t schedule_hip_stream();

  // Return the last HIP stream used.
  hipStream_t last_hip_stream();

  CommandEncoder& get_encoder();

  Device& device() {
    return device_;
  }

 private:
  Device& device_;
  HipStream stream_;
  std::unique_ptr<CommandEncoder> encoder_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current HIP device, required by some HIP calls.
  void make_current();

  DeviceStream& get_stream(Stream s);

  int hip_device() const {
    return device_;
  }
  int compute_capability_major() const {
    return compute_capability_major_;
  }
  int compute_capability_minor() const {
    return compute_capability_minor_;
  }
  rocblas_handle rocblas_handle() const {
    return rocblas_handle_;
  }

 private:
  int device_;
  int compute_capability_major_;
  int compute_capability_minor_;
  rocblas_handle rocblas_handle_;
  std::unordered_map<int, DeviceStream> streams_;
};

class CommandEncoder {
 public:
  explicit CommandEncoder(DeviceStream& stream);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  void set_input_array(const array& arr) {}
  void set_output_array(const array& arr) {}

  void add_temporary(const array& arr) {
    temporaries_.push_back(arr.data_shared_ptr());
  }

  void add_completed_handler(std::function<void()> task);
  void end_encoding();
  void commit();

  // Schedule a HIP stream for |fun| to launch kernels, and check error
  // afterwards.
  template <typename F>
  void launch_kernel(F&& fun) {
    launch_kernel(stream_.schedule_hip_stream(), std::forward<F>(fun));
  }

  template <typename F>
  void launch_kernel(hipStream_t stream, F&& fun) {
    device_.make_current();
    fun(stream);
    check_hip_error("kernel launch", hipGetLastError());
    has_gpu_work_ = true;
  }

  Device& device() {
    return device_;
  }

  DeviceStream& stream() {
    return stream_;
  }

  bool has_gpu_work() const {
    return has_gpu_work_;
  }

 private:
  Device& device_;
  DeviceStream& stream_;
  Worker worker_;
  bool has_gpu_work_{false};
  std::vector<std::shared_ptr<array::Data>> temporaries_;
};

Device& device(mlx::core::Device device);
DeviceStream& get_stream(Stream s);
CommandEncoder& get_command_encoder(Stream s);

// Utility function to check HIP errors
void check_hip_error(const char* msg, hipError_t error);

} // namespace rocm

} // namespace mlx::core