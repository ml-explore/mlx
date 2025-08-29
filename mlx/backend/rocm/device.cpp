// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/backend/rocm/worker.h"

#include <fmt/format.h>

namespace mlx::core {

namespace rocm {

DeviceStream::DeviceStream(Device& device) : device_(device), stream_(device) {}

void DeviceStream::synchronize() {
  CHECK_HIP_ERROR(hipStreamSynchronize(stream_));
}

hipStream_t DeviceStream::schedule_hip_stream() {
  // TODO: Return a stream that maximizes parallelism.
  return stream_;
}

hipStream_t DeviceStream::last_hip_stream() {
  return stream_;
}

CommandEncoder& DeviceStream::get_encoder() {
  if (!encoder_) {
    encoder_ = std::make_unique<CommandEncoder>(*this);
  }
  return *encoder_;
}

Device::Device(int device) : device_(device) {
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &compute_capability_major_,
      hipDeviceAttributeComputeCapabilityMajor,
      device_));
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &compute_capability_minor_,
      hipDeviceAttributeComputeCapabilityMinor,
      device_));

  // Validate device requirements
  int attr = 0;
  CHECK_HIP_ERROR(hipDeviceGetAttribute(
      &attr, hipDeviceAttributeConcurrentManagedAccess, device_));
  if (attr != 1) {
    // ROCm unified memory might not be available on all devices
    // This is a warning rather than an error for ROCm
    // TODO: Add proper ROCm unified memory checking
  }

  // Create rocBLAS handle
  make_current();
  CHECK_HIP_ERROR(
      static_cast<hipError_t>(rocblas_create_handle(&rocblas_handle_)));
}

Device::~Device() {
  if (rocblas_handle_) {
    rocblas_destroy_handle(rocblas_handle_);
  }
}

void Device::make_current() {
  // Cache current device to reduce HIP API calls
  static int current = 0;
  if (current != device_) {
    CHECK_HIP_ERROR(hipSetDevice(device_));
    current = device_;
  }
}

DeviceStream& Device::get_stream(Stream s) {
  auto it = streams_.find(s.index);
  if (it == streams_.end()) {
    it = streams_.try_emplace(s.index, *this).first;
  }
  return it->second;
}

CommandEncoder::CommandEncoder(DeviceStream& s)
    : device_(s.device()), stream_(s) {}

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_.add_task(std::move(task));
}

void CommandEncoder::end_encoding() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }

  // There is no kernel running, run completion handlers immediately.
  if (!has_gpu_work_) {
    worker_.consume_in_this_thread();
    return;
  }
  has_gpu_work_ = false;

  // Commit tasks
  commit();
}

void CommandEncoder::commit() {
  worker_.commit(stream_.last_hip_stream());
}

Device& device(mlx::core::Device device) {
  static std::unordered_map<int, Device> devices;
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

DeviceStream& get_stream(Stream s) {
  return device(s.device).get_stream(s);
}

CommandEncoder& get_command_encoder(Stream s) {
  return get_stream(s).get_encoder();
}

} // namespace rocm

} // namespace mlx::core