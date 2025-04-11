// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/stream.h"

#include <cublasLt.h>
#include <thrust/execution_policy.h>

#include <unordered_map>

namespace mlx::core::mxcuda {

class Device;
class CommandEncoder;

// A stream in MLX consists of multiple CUDA stream.
class DeviceStream {
 public:
  DeviceStream(Device& device, Stream stream);
  ~DeviceStream();

  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  // Wait until all current tasks finish.
  void synchronize();

  // Return a CUDA stream for launching kernels.
  cudaStream_t schedule_cuda_stream();

  // Return the last stream used.
  cudaStream_t last_cuda_stream();

  // Push a function that will run after current eval ends.
  void add_cleanup(std::function<void()> func);

  // Run the cleanup callbacks after current tasks end.
  void finalize();

  CommandEncoder& get_encoder();

  Device& device() {
    return device_;
  }

 private:
  // Run the function in host after last launched work finishes. This call adds
  // at least 20µs latency in cuda stream, so only use it when necessary.
  void add_host_callback(std::function<void()> func);

  Device& device_;
  cudaStream_t stream_;
  std::unique_ptr<CommandEncoder> encoder_;
  std::vector<std::function<void()>> cleanups_;
};

class Device {
 public:
  explicit Device(int device);
  ~Device();

  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;

  // Make this instance the current CUDA device, required by some CUDA calls.
  void make_current();

  DeviceStream& get_stream(Stream stream);

  int cuda_device() const {
    return device_;
  }

  cublasLtHandle_t lt_handle() const {
    return lt_;
  }

 private:
  int device_;
  cublasLtHandle_t lt_;
  std::unordered_map<int, DeviceStream> streams_;
};

class CommandEncoder {
 public:
  explicit CommandEncoder(DeviceStream& stream);

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_input_array(const Arrays&... arrays) {}

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_output_array(const Arrays&... arrays) {}

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void add_temporary(Arrays&&... arrays) {
    (temporaries_.push_back(std::forward<Arrays>(arrays)), ...);
  }

  template <typename F>
  void launch_kernel(F&& fun) {
    launch_kernel_with(std::forward<F>(fun), stream_.schedule_cuda_stream());
  }

  template <typename F>
  void launch_kernel_sequencially(F&& fun) {
    launch_kernel_with(std::forward<F>(fun), stream_.last_cuda_stream());
  }

  template <typename F>
  void launch_thrust(F&& fun) {
    launch_kernel([&](cudaStream_t stream) {
      // Make thrust dispatch work on stream asynchronously.
      // TODO: If we are going to keep the thrust APIs in the end, we should
      // use a custom allocator that works with existing buffer cache.
      auto nosync_exec_policy = thrust::cuda::par_nosync.on(stream);
      fun(nosync_exec_policy);
    });
  }

  Device& device() {
    return device_;
  }

  DeviceStream& stream() {
    return stream_;
  }

 private:
  template <typename F>
  void launch_kernel_with(F&& fun, cudaStream_t stream) {
    device_.make_current();
    fun(stream);
    check_cuda_error("kernel launch", cudaGetLastError());
    if (!temporaries_.empty()) {
      stream_.add_cleanup([temporaries = std::move(temporaries_)]() {});
    }
  }

  Device& device_;
  DeviceStream& stream_;
  std::vector<array> temporaries_;
};

Device& device(mlx::core::Device device);
DeviceStream& get_stream(Stream stream);
CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::mxcuda
