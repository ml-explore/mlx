// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"
#include "mlx/fence.h"
#include "mlx/scheduler.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace {

__host__ __device__ void busy_wait(cuda::atomic<uint64_t>* ac, uint64_t value) {
  while (true) {
    // In theory the atomic_thread_fence is not needed, but for CUDA 11 without
    // it the load() may never return new value.
    cuda::atomic_thread_fence(cuda::memory_order_seq_cst);
    uint64_t current = ac->load();
    if (current >= value) {
      break;
    }
  }
}

__global__ void busy_wait_kernel(cuda::atomic<uint64_t>* ac, uint64_t value) {
  busy_wait(ac, value);
}

} // namespace

struct FenceImpl {
  uint32_t count;
  cu::SharedEvent event;
};

Fence::Fence(Stream s) {
  fence_ = std::shared_ptr<void>(
      new FenceImpl{0}, [](void* ptr) { delete static_cast<FenceImpl*>(ptr); });
}

void Fence::wait(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  // We can't use SharedEvent::wait because it could hang in CUDA 11, see also:
  // https://github.com/ml-explore/mlx/issues/2137
  const auto& ac = fence->event.atomic();
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [ac, count = fence->count]() {
      nvtx3::scoped_range r("Fence::wait()");
      busy_wait(ac.get(), count);
    });
  } else {
    nvtx3::scoped_range r("Fence::wait(s)");
    auto& encoder = cu::get_command_encoder(s);
    encoder.launch_kernel(
        encoder.stream().last_cuda_stream(), [&](cudaStream_t stream) {
          busy_wait_kernel<<<1, 1, 0>>>(ac.get(), fence->count);
        });
    encoder.add_completed_handler([ac]() {});
    encoder.end_encoding();
  }
}

void Fence::update(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  fence->event.signal(s, fence->count);
}

} // namespace mlx::core
