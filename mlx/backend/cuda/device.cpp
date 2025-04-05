// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/backend/metal/metal.h"

#include <fmt/format.h>

namespace mlx::core {

namespace mxcuda {

DeviceStream::DeviceStream(Device& device, Stream stream) : device_(device) {
  device_.make_current();
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
}

DeviceStream::~DeviceStream() {
  CHECK_CUDA_ERROR(cudaStreamDestroy(stream_));
}

void DeviceStream::synchronize() {
  // TODO: Wait for all cuda streams in mlx stream.
  cudaStreamSynchronize(stream_);
}

cudaStream_t DeviceStream::schedule_cuda_stream() {
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

CommandEncoder& DeviceStream::get_encoder() {
  if (!encoder_) {
    encoder_ = std::make_unique<CommandEncoder>(*this);
  }
  return *encoder_;
}

Device::Device(int device) : device_(device) {
  // Validate the requirements of device.
  int attr = 0;
  cudaDeviceGetAttribute(&attr, cudaDevAttrConcurrentManagedAccess, device_);
  if (attr != 1) {
    throw std::runtime_error(fmt::format(
        "Device {} does not support synchronization in managed memory.",
        device_));
  }
  // The cublasLt handle is used for matmul.
  // TODO: Allocate a workspace buffer for cublas.
  make_current();
  cublasLtCreate(&lt_);
}

Device::~Device() {
  cublasLtDestroy(lt_);
}

void Device::make_current() {
  // We need to set/get current CUDA device very frequently, cache it to reduce
  // actual calls of CUDA APIs. This function assumes single-thread in host.
  static int current = 0;
  if (current != device_) {
    CHECK_CUDA_ERROR(cudaSetDevice(device_));
    current = device_;
  }
}

DeviceStream& Device::get_stream(Stream stream) {
  auto it = streams_.find(stream.index);
  if (it == streams_.end()) {
    it = streams_.try_emplace(stream.index, *this, stream).first;
  }
  return it->second;
}

CommandEncoder::CommandEncoder(DeviceStream& stream)
    : device_(stream.device()), stream_(stream) {}

void CommandEncoder::prefetch_memory(const array& arr) {
  // TODO: Profile whether prefetching the whole buffer would be faster.
  const void* data = arr.data<void>();
  size_t size = arr.data_size() * arr.itemsize();
  if (data && size > 0) {
    // TODO: Use a stream that maximizes parallelism.
    CHECK_CUDA_ERROR(cudaMemPrefetchAsync(
        data, size, device_.cuda_device(), stream_.last_cuda_stream()));
  }
}

Device& device(mlx::core::Device device) {
  static std::unordered_map<int, Device> devices;
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

DeviceStream& get_stream(Stream stream) {
  return device(stream.device).get_stream(stream);
}

CommandEncoder& get_command_encoder(Stream stream) {
  return get_stream(stream).get_encoder();
}

} // namespace mxcuda

namespace metal {

void new_stream(Stream stream) {
  // Ensure the static stream objects get created.
  mxcuda::get_command_encoder(stream);
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info() {
  throw std::runtime_error(
      "[metal::device_info] Not implemented in CUDA backend.");
};

} // namespace metal

} // namespace mlx::core
