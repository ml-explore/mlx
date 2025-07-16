// Copyright © 2024 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"
#include "mlx/backend/cuda/utils.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

///////////////////////////////////////////////////////////////////////////////
// CudaEvent implementations
///////////////////////////////////////////////////////////////////////////////

// Cuda event managed with RAII.
class CudaEventHandle {
 public:
  CudaEventHandle() {
    CHECK_CUDA_ERROR(cudaEventCreateWithFlags(
        &event_, cudaEventDisableTiming | cudaEventBlockingSync));
  }

  ~CudaEventHandle() {
    CHECK_CUDA_ERROR(cudaEventDestroy(event_));
  }

  CudaEventHandle(const CudaEventHandle&) = delete;
  CudaEventHandle& operator=(const CudaEventHandle&) = delete;

  operator cudaEvent_t() const {
    return event_;
  }

 private:
  cudaEvent_t event_;
};

CudaEvent::CudaEvent() : event_(std::make_shared<CudaEventHandle>()) {}

void CudaEvent::wait() {
  nvtx3::scoped_range r("cu::CudaEvent::wait");
  if (!recorded_) {
    throw std::runtime_error("Should not wait on a CudaEvent before record.");
  }
  cudaEventSynchronize(*event_);
}

void CudaEvent::wait(cudaStream_t stream) {
  if (!recorded_) {
    throw std::runtime_error("Should not wait on a CudaEvent before record.");
  }
  cudaStreamWaitEvent(stream, *event_);
}

void CudaEvent::wait(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this]() mutable { wait(); });
  } else {
    auto& enc = cu::get_command_encoder(s);
    enc.commit();
    wait(enc.stream());
  }
}

void CudaEvent::record(cudaStream_t stream) {
  cudaEventRecord(*event_, stream);
  recorded_ = true;
}

void CudaEvent::record(Stream s) {
  if (s.device == mlx::core::Device::cpu) {
    throw std::runtime_error("CudaEvent can not wait on cpu stream.");
  } else {
    auto& enc = cu::get_command_encoder(s);
    enc.commit();
    record(enc.stream());
  }
}

bool CudaEvent::completed() const {
  return cudaEventQuery(*event_) == cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////
// SharedEvent implementations
///////////////////////////////////////////////////////////////////////////////

__host__ __device__ void event_wait(SharedEvent::Atomic* ac, uint64_t value) {
  uint64_t current;
  while ((current = ac->load()) < value) {
    ac->wait(current);
  }
}

__host__ __device__ void event_signal(SharedEvent::Atomic* ac, uint64_t value) {
  ac->store(value);
  ac->notify_all();
}

__global__ void event_wait_kernel(SharedEvent::Atomic* ac, uint64_t value) {
  event_wait(ac, value);
}

__global__ void event_signal_kernel(SharedEvent::Atomic* ac, uint64_t value) {
  event_signal(ac, value);
}

SharedEvent::SharedEvent() {
  // Allocate cuda::atomic on managed memory.
  Atomic* ac;
  CHECK_CUDA_ERROR(cudaMallocManaged(&ac, sizeof(Atomic)));
  new (ac) Atomic(0);
  ac_ = std::shared_ptr<Atomic>(ac, [](Atomic* ptr) {
    ptr->~Atomic();
    allocator().cuda_free(ptr);
  });
}

void SharedEvent::wait(uint64_t value) {
  nvtx3::scoped_range r("cu::SharedEvent::wait");
  event_wait(ac_.get(), value);
}

void SharedEvent::wait(cudaStream_t stream, uint64_t value) {
  event_wait_kernel<<<1, 1, 0, stream>>>(ac_.get(), value);
}

void SharedEvent::wait(Stream s, uint64_t value) {
  nvtx3::scoped_range r("cu::SharedEvent::wait(s)");
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { wait(value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    wait(encoder.stream(), value);
    encoder.add_completed_handler([ac = ac_]() {});
  }
}

void SharedEvent::signal(uint64_t value) {
  nvtx3::scoped_range r("cu::SharedEvent::signal");
  event_signal(ac_.get(), value);
}

void SharedEvent::signal(cudaStream_t stream, uint64_t value) {
  event_signal_kernel<<<1, 1, 0, stream>>>(ac_.get(), value);
}

void SharedEvent::signal(Stream s, uint64_t value) {
  nvtx3::scoped_range r("cu::SharedEvent::signal(s)");
  if (s.device == mlx::core::Device::cpu) {
    // Signal through a GPU stream so the atomic is updated in GPU - updating
    // the atomic in CPU sometimes does not get GPU notified.
    static CudaStream stream(device(mlx::core::Device::gpu));
    scheduler::enqueue(s, [*this, value]() mutable { signal(stream, value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    signal(encoder.stream(), value);
    encoder.add_completed_handler([ac = ac_]() {});
  }
}

bool SharedEvent::is_signaled(uint64_t value) const {
  nvtx3::scoped_range r("cu::SharedEvent::is_signaled");
  return ac_->load() >= value;
}

uint64_t SharedEvent::value() const {
  nvtx3::scoped_range r("cu::SharedEvent::value");
  return ac_->load();
}

} // namespace cu

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

struct EventImpl {
  // CudaEvent is preferred when possible because it is fast, however we have
  // to fallback to SharedEvent in following cases:
  // 1. the event is used to wait/signal a cpu stream;
  // 2. signal value other than 1 has been specified.
  std::unique_ptr<cu::CudaEvent> cuda;
  std::unique_ptr<cu::SharedEvent> shared;

  bool is_created() const {
    return cuda || shared;
  }

  void ensure_created(Stream s, uint64_t signal_value) {
    if (is_created()) {
      return;
    }
    if (s.device == mlx::core::Device::cpu || signal_value > 1) {
      nvtx3::mark("Using slow SharedEvent");
      shared = std::make_unique<cu::SharedEvent>();
    } else {
      cuda = std::make_unique<cu::CudaEvent>();
    }
  }
};

} // namespace

Event::Event(Stream s) : stream_(s) {
  event_ = std::shared_ptr<void>(
      new EventImpl(), [](void* ptr) { delete static_cast<EventImpl*>(ptr); });
}

void Event::wait() {
  auto* event = static_cast<EventImpl*>(event_.get());
  assert(event->is_created());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->wait();
  } else {
    event->shared->wait(value());
  }
}

void Event::wait(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  assert(event->is_created());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->wait(s);
  } else {
    event->shared->wait(s, value());
  }
}

void Event::signal(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  event->ensure_created(s, value());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->record(s);
  } else {
    event->shared->signal(s, value());
  }
}

bool Event::is_signaled() const {
  auto* event = static_cast<EventImpl*>(event_.get());
  if (!event->is_created()) {
    return false;
  }
  if (event->cuda) {
    assert(value() == 1);
    return event->cuda->recorded() && event->cuda->completed();
  } else {
    return event->shared->is_signaled(value());
  }
}

} // namespace mlx::core
