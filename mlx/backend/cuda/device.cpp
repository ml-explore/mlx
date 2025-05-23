// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/worker.h"
#include "mlx/backend/metal/metal.h"

#include <fmt/format.h>
#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

DeviceStream::DeviceStream(Device& device) : device_(device), stream_(device) {}

void DeviceStream::synchronize() {
  cudaStreamSynchronize(stream_);
}

cudaStream_t DeviceStream::schedule_cuda_stream() {
  // TODO: Return a stream that maximizes parallelism.
  return stream_;
}

cudaStream_t DeviceStream::last_cuda_stream() {
  return stream_;
}

CommandEncoder& DeviceStream::get_encoder() {
  if (!encoder_) {
    encoder_ = std::make_unique<CommandEncoder>(*this);
  }
  return *encoder_;
}

Device::Device(int device) : device_(device) {
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &compute_capability_major_, cudaDevAttrComputeCapabilityMajor, device_));
  // Validate the requirements of device.
  int attr = 0;
  CHECK_CUDA_ERROR(cudaDeviceGetAttribute(
      &attr, cudaDevAttrConcurrentManagedAccess, device_));
  if (attr != 1) {
    throw std::runtime_error(fmt::format(
        "Device {} does not support synchronization in managed memory.",
        device_));
  }
  // The cublasLt handle is used by matmul.
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

  // Put completion handlers in a batch.
  worker_.end_batch();

  // Signaling kernel completion is expensive, delay until enough batches.
  // TODO: This number is arbitrarily picked, profile for a better stragety.
  if (worker_.uncommited_batches() > 8) {
    commit();
  }
}

void CommandEncoder::commit() {
  worker_.commit(stream_.last_cuda_stream());
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

} // namespace cu

} // namespace mlx::core
