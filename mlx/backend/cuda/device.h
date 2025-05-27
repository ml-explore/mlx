// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/stream.h"

#include <thrust/execution_policy.h>

#include <unordered_map>

namespace mlx::core::cu {

class Device;
class CommandEncoder;

class DeviceStream {
 public:
  explicit DeviceStream(Device& device);

  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  // Wait until kernels in the stream complete.
  void synchronize();

  // Return a cuda stream for launching kernels.
  cudaStream_t schedule_cuda_stream();

  // Return the last cuda stream used.
  cudaStream_t last_cuda_stream();

  CommandEncoder& get_encoder();

  Device& device() {
    return device_;
  }

 private:
  Device& device_;
  CudaStream stream_;
  std::unique_ptr<CommandEncoder> encoder_;
};

class Device {
 public:
  explicit Device(int device);

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this device the current cuda device, required by some cuda calls.
  void make_current();

  DeviceStream& get_stream(Stream s);

  int cuda_device() const {
    return device_;
  }

 private:
  int device_;
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

  // Schedule a cuda stream for |fun| to launch kernels, and check error
  // afterwards.
  template <typename F>
  void launch_kernel(F&& fun) {
    launch_kernel(stream_.schedule_cuda_stream(), std::forward<F>(fun));
  }

  template <typename F>
  void launch_kernel(cudaStream_t stream, F&& fun) {
    device_.make_current();
    fun(stream);
    check_cuda_error("kernel launch", cudaGetLastError());
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

// Return an execution policy that does not sync for result.
// Note that not all thrust APIs support async policy, confirm before using.
inline auto thrust_policy(cudaStream_t stream) {
  // TODO: Connect thrust's custom allocator with mlx's allocator.
  return thrust::cuda::par_nosync.on(stream);
}

} // namespace mlx::core::cu
