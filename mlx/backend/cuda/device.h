// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/stream.h"

namespace mlx::core::mxcuda {

// We have to set current device before calling some APIs to make multi-device
// work, including kernel launching.
inline void set_cuda_device(Device device) {
  thread_local static int device_ = 0;
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
  void set_input_array(const Arrays&... arrays) {}
  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void set_output_array(const Arrays&... arrays) {}
  template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
  void add_temporary(Arrays&&... arrays) {
    (temporaries_.push_back(std::forward<Arrays>(arrays)), ...);
  }

  template <class F>
  void launch_kernel(F&& fun) {
    fun(stream_.schedule_cuda_stream());
    stream_.add_host_callback([temporaries = std::move(temporaries_)]() {});
  }

  DeviceStream& stream() {
    return stream_;
  }

 private:
  DeviceStream stream_;
  std::vector<array> temporaries_;
};

DeviceStream& get_stream(Stream stream);
CommandEncoder& get_command_encoder(Stream stream);

} // namespace mlx::core::mxcuda
