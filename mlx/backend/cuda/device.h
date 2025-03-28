// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/stream.h"

#include <thrust/execution_policy.h>

namespace mlx::core::mxcuda {

// We need to set/get current CUDA device very frequently, cache it to reduce
// actual calls of CUDA APIs. This function assumes single-thread in host.
inline void set_cuda_device(Device device) {
  static int device_ = 0;
  if (device.index != device_) {
    CHECK_CUDA_ERROR(cudaSetDevice(device.index));
    device_ = device.index;
  }
}

// Get the maxThreadsPerBlock property of device.
size_t max_threads_per_block(Device device);

// A stream in MLX consists of multiple CUDA stream.
class DeviceStream {
 public:
  explicit DeviceStream(Stream stream);
  ~DeviceStream();

  // Return a CUDA stream for launching kernels.
  cudaStream_t schedule_cuda_stream();

  // Return the last stream used.
  cudaStream_t last_cuda_stream();

  // Run the function in host after last launched work finishes.
  void add_host_callback(std::function<void()> func);

  const Device& device() const {
    return device_;
  }

 private:
  Device device_;
  cudaStream_t stream_;
};

class CommandEncoder {
 public:
  explicit CommandEncoder(Stream stream) : stream_(stream) {}

  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_input_array(const Arrays&... arrays) {
    (prefetch_memory(arrays), ...);
  }

  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_output_array(const Arrays&... arrays) {
    (prefetch_memory(arrays), ...);
  }

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

  DeviceStream& stream() {
    return stream_;
  }

 private:
  template <typename F>
  void launch_kernel_with(F&& fun, cudaStream_t stream) {
    set_cuda_device(stream_.device());
    fun(stream);
    check_cuda_error("kernel launch", cudaGetLastError());
    if (!temporaries_.empty()) {
      stream_.add_host_callback([temporaries = std::move(temporaries_)]() {});
    }
  }

  void prefetch_memory(const array& arr);

  DeviceStream stream_;
  std::vector<array> temporaries_;
};

DeviceStream& get_stream(Stream stream);
CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::mxcuda
