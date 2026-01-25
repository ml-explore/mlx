// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/worker.h"
#include "mlx/backend/rocm/utils.h"
#include "mlx/utils.h"

#include <future>
#include <sstream>

namespace mlx::core::rocm {

namespace {

// Can be tuned with MLX_MAX_OPS_PER_BUFFER
constexpr int default_max_ops_per_buffer = 20;

} // namespace

Device::Device(int device) : device_(device) {
  make_current();
  CHECK_ROCBLAS_ERROR(rocblas_create_handle(&rocblas_));
}

Device::~Device() {
  if (rocblas_) {
    rocblas_destroy_handle(rocblas_);
  }
}

void Device::make_current() {
  // We need to set/get current HIP device very frequently, cache it to reduce
  // actual calls of HIP APIs. This function assumes single-thread in host.
  static int current = -1;
  if (current != device_) {
    CHECK_HIP_ERROR(hipSetDevice(device_));
    current = device_;
  }
}

CommandEncoder& Device::get_command_encoder(Stream s) {
  auto it = encoders_.find(s.index);
  if (it == encoders_.end()) {
    auto [inserted_it, success] = encoders_.emplace(s.index, std::make_unique<CommandEncoder>(*this));
    it = inserted_it;
  }
  return *it->second;
}

CommandEncoder::CommandEncoder(Device& d)
    : device_(d), stream_(d), worker_(std::make_unique<Worker>()) {}

CommandEncoder::~CommandEncoder() = default;

void CommandEncoder::add_completed_handler(std::function<void()> task) {
  worker_->add_task(std::move(task));
}

void CommandEncoder::set_input_array(const array& arr) {
  // For now, no-op - can be used for dependency tracking
}

void CommandEncoder::set_output_array(const array& arr) {
  // For now, no-op - can be used for dependency tracking
}

void CommandEncoder::maybe_commit() {
  if (node_count_ >= env::max_ops_per_buffer(default_max_ops_per_buffer)) {
    commit();
  }
}

void CommandEncoder::commit() {
  if (!temporaries_.empty()) {
    add_completed_handler([temporaries = std::move(temporaries_)]() {});
  }
  node_count_ = 0;
  
  // Put completion handlers in a batch.
  worker_->commit(stream_);
}

void CommandEncoder::synchronize() {
  hipStreamSynchronize(stream_);
  auto p = std::make_shared<std::promise<void>>();
  std::future<void> f = p->get_future();
  add_completed_handler([p = std::move(p)]() { p->set_value(); });
  commit();
  f.wait();
}

Device& device(mlx::core::Device device) {
  static std::unordered_map<int, Device> devices;
  auto it = devices.find(device.index);
  if (it == devices.end()) {
    it = devices.try_emplace(device.index, device.index).first;
  }
  return it->second;
}

CommandEncoder& get_command_encoder(Stream s) {
  return device(s.device).get_command_encoder(s);
}

} // namespace mlx::core::rocm
