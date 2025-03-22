// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/metal/metal.h"

#include <cuda/atomic>
#include <unordered_map>

namespace mlx::core {

namespace mxcuda {

size_t max_threads_per_block(Device device) {
  cudaDeviceProp prop;
  CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, device.index));
  return prop.maxThreadsPerBlock;
}

DeviceStream::DeviceStream(Stream stream) : device_(stream.device) {
  set_cuda_device(device_);
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
  // Validate the requirements of device.
  // TODO: Validate per-device instead of per-stream.
  int a = 0;
  cudaDeviceGetAttribute(&a, cudaDevAttrConcurrentManagedAccess, device_.index);
  if (a != 1) {
    throw std::runtime_error(
        "Synchronization between CPU/GPU not supported for managed memory.");
  }
}

DeviceStream::~DeviceStream() {
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

cudaStream_t DeviceStream::schedule_cuda_stream() {
  set_cuda_device(device_);
  // TODO: Return a stream that maximizes parallelism.
  return stream_;
}

cudaStream_t DeviceStream::last_cuda_stream() {
  return stream_;
}

void DeviceStream::add_host_callback(std::function<void()> func) {
  CHECK_CUDA_ERROR(cudaLaunchHostFunc(
      last_cuda_stream(),
      [](void* ptr) {
        auto* func = static_cast<std::function<void()>*>(ptr);
        (*func)();
        delete func;
      },
      new std::function<void()>(std::move(func))));
}

DeviceStream& get_stream(Stream stream) {
  return get_command_encoder(stream).stream();
}

CommandEncoder& get_command_encoder(Stream stream) {
  static std::unordered_map<int, CommandEncoder> encoder_map;
  auto it = encoder_map.find(stream.index);
  if (it == encoder_map.end()) {
    it = encoder_map.emplace(stream.index, stream).first;
  }
  return it->second;
}

} // namespace mxcuda

namespace metal {

void new_stream(Stream stream) {
  // Ensure the static stream objects are created.
  mxcuda::get_command_encoder(stream);
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Not implemented in CUDA backend.");
};

std::unique_ptr<void, std::function<void(void*)>> new_scoped_memory_pool() {
  return nullptr;
}

} // namespace metal

} // namespace mlx::core
