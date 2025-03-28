// Copyright © 2024 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <cuda/atomic>

namespace mlx::core {

namespace {

__host__ __device__ void event_wait(
    cuda::atomic<uint64_t>* ac,
    uint64_t value) {
  uint64_t current;
  while ((current = ac->load()) < value) {
    ac->wait(current);
  }
}

__host__ __device__ void event_signal(
    cuda::atomic<uint64_t>* ac,
    uint64_t value) {
  ac->store(value);
  ac->notify_all();
}

__global__ void event_wait_kernel(cuda::atomic<uint64_t>* ac, uint64_t value) {
  event_wait(ac, value);
}

__global__ void event_signal_kernel(
    cuda::atomic<uint64_t>* ac,
    uint64_t value) {
  event_signal(ac, value);
}

} // namespace

Event::Event(Stream stream) : stream_(stream) {
  // Allocate cuda::atomic on managed memory.
  cuda::atomic<uint64_t>* ac;
  CHECK_CUDA_ERROR(cudaMallocManaged(&ac, sizeof(cuda::atomic<uint64_t>)));
  new (ac) std::atomic<uint64_t>(0);
  // Store it in a shared_ptr.
  auto dtor = [](void* ptr) {
    static_cast<cuda::atomic<uint64_t>*>(ptr)->~atomic<uint64_t>();
    cudaFree(ptr);
  };
  event_ = std::shared_ptr<void>(ac, dtor);
}

void Event::wait() {
  auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
  event_wait(ac, value());
}

void Event::signal() {
  auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
  event_signal(ac, value());
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [this]() mutable { wait(); });
  } else {
    mxcuda::get_command_encoder(stream).launch_kernel_sequencially(
        [this](cudaStream_t s) {
          auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
          event_wait_kernel<<<1, 1, 0, s>>>(ac, value());
        });
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [this]() mutable { signal(); });
  } else {
    mxcuda::get_command_encoder(stream).launch_kernel_sequencially(
        [this](cudaStream_t s) {
          auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
          event_signal_kernel<<<1, 1, 0, s>>>(ac, value());
        });
  }
}

bool Event::is_signaled() const {
  auto* ac = static_cast<cuda::atomic<uint64_t>*>(event_.get());
  return ac->load() >= value();
}
} // namespace mlx::core
