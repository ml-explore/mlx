// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/utils.h"

namespace mlx::core::rocm {

DeviceStream::DeviceStream(Device& device) : device_(device) {
  check_hip_error("hipStreamCreate", hipStreamCreate(&stream_));
  encoder_ = std::make_unique<CommandEncoder>(*this);
}

void DeviceStream::synchronize() {
  check_hip_error("hipStreamSynchronize", hipStreamSynchronize(stream_));
}

hipStream_t DeviceStream::schedule_hip_stream() {
  return stream_;
}

hipStream_t DeviceStream::last_hip_stream() {
  return stream_;
}

CommandEncoder& DeviceStream::get_encoder() {
  return *encoder_;
}

Device::Device(int device) : device_(device) {
  check_hip_error("hipSetDevice", hipSetDevice(device_));

  // Get device properties
  hipDeviceProp_t prop;
  check_hip_error(
      "hipGetDeviceProperties", hipGetDeviceProperties(&prop, device_));
  compute_capability_major_ = prop.major;
  compute_capability_minor_ = prop.minor;

  // Create rocBLAS handle
  check_hip_error(
      "rocblas_create_handle",
      static_cast<hipError_t>(rocblas_create_handle(&rocblas_handle_)));
}

Device::~Device() {
  if (rocblas_handle_) {
    rocblas_destroy_handle(rocblas_handle_);
  }
}

void Device::make_current() {
  check_hip_error("hipSetDevice", hipSetDevice(device_));
}

DeviceStream& Device::get_stream(Stream s) {
  auto it = streams_.find(s.index);
  if (it != streams_.end()) {
    return it->second;
  }

  auto [new_it, inserted] = streams_.emplace(s.index, DeviceStream(*this));
  return new_it->second;
}

CommandEncoder::CommandEncoder(DeviceStream& stream)
    : device_(stream.device()), stream_(stream), worker_() {}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_.enqueue(task);
}

void CommandEncoder::end_encoding() {
  // Implementation for ending encoding
}

void CommandEncoder::commit() {
  worker_.commit();
}

// Global device management
static std::unordered_map<int, std::unique_ptr<Device>> devices_;

Device& device(mlx::core::Device device) {
  auto it = devices_.find(device.index);
  if (it != devices_.end()) {
    return *it->second;
  }

  auto new_device = std::make_unique<Device>(device.index);
  Device& dev_ref = *new_device;
  devices_[device.index] = std::move(new_device);
  return dev_ref;
}

DeviceStream& get_stream(Stream s) {
  // Use default device (index 0) for now
  return device(mlx::core::Device{mlx::core::Device::gpu, 0}).get_stream(s);
}

CommandEncoder& get_command_encoder(Stream s) {
  return get_stream(s).get_encoder();
}

} // namespace mlx::core::rocm