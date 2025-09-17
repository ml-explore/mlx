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

namespace {

// Manage cached cudaEvent_t objects.
struct CudaEventFactory {
  static RawCudaEvent create(int flags) {
    auto& cache = cache_for(flags);
    if (cache.empty()) {
      return RawCudaEvent(flags);
    } else {
      RawCudaEvent ret = std::move(cache.back());
      cache.pop_back();
      return ret;
    }
  }

  static void release(RawCudaEvent event) {
    assert(event != nullptr);
    cache_for(event.flags).push_back(std::move(event));
  }

  static std::vector<RawCudaEvent>& cache_for(int flags) {
    static std::map<int, std::vector<RawCudaEvent>> cache;
    return cache[flags];
  }
};

} // namespace

RawCudaEvent::RawCudaEvent(int flags) : flags(flags) {
  CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&handle_, flags));
}

CudaEvent::CudaEvent(int flags) : event_(CudaEventFactory::create(flags)) {
  assert(event_ != nullptr);
}

CudaEvent::~CudaEvent() {
  if (event_) {
    CudaEventFactory::release(std::move(event_));
  }
}

void CudaEvent::wait() {
  nvtx3::scoped_range r("cu::CudaEvent::wait");
  assert(event_ != nullptr);
  cudaEventSynchronize(event_);
}

void CudaEvent::wait(cudaStream_t stream) {
  assert(event_ != nullptr);
  cudaStreamWaitEvent(stream, event_);
}

void CudaEvent::record(cudaStream_t stream) {
  assert(event_ != nullptr);
  cudaEventRecord(event_, stream);
}

bool CudaEvent::completed() const {
  assert(event_ != nullptr);
  return cudaEventQuery(event_) == cudaSuccess;
}

// Wraps CudaEvent with a few features:
// 1. The class can be copied.
// 2. Make wait/record work with CPU streams.
// 3. Add checks for waiting on un-recorded event.
class CudaEventWrapper {
 public:
  CudaEventWrapper()
      : event_(std::make_shared<CudaEvent>(
            cudaEventDisableTiming | cudaEventBlockingSync)) {}

  void wait() {
    event_->wait();
  }

  void wait(Stream s) {
    if (s.device == mlx::core::Device::cpu) {
      scheduler::enqueue(s, [*this]() mutable {
        check_recorded();
        event_->wait();
      });
    } else {
      check_recorded();
      auto& encoder = cu::get_command_encoder(s);
      encoder.commit();
      event_->wait(encoder.stream());
    }
  }

  void record(Stream s) {
    if (s.device == mlx::core::Device::cpu) {
      throw std::runtime_error("CudaEvent can not wait on CPU stream.");
    } else {
      auto& encoder = cu::get_command_encoder(s);
      encoder.commit();
      event_->record(encoder.stream());
      recorded_ = true;
    }
  }

  bool is_signaled() const {
    return recorded_ && event_->completed();
  }

 private:
  void check_recorded() const {
    if (!recorded_) {
      throw std::runtime_error(
          "Should not wait on a CudaEvent before recording.");
    }
  }

  std::shared_ptr<CudaEvent> event_;
  bool recorded_{false};
};

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
  buf_ = std::shared_ptr<Buffer>(
      new Buffer{allocator().malloc(sizeof(Atomic))}, [](Buffer* ptr) {
        allocator().free(*ptr);
        delete ptr;
      });
  *static_cast<uint64_t*>(buf_->raw_ptr()) = 0;
}

void SharedEvent::wait(uint64_t value) {
  nvtx3::scoped_range r("cu::SharedEvent::wait");
  event_wait(atomic(), value);
}

void SharedEvent::wait(cudaStream_t stream, uint64_t value) {
  event_wait_kernel<<<1, 1, 0, stream>>>(atomic(), value);
}

void SharedEvent::wait(Stream s, uint64_t value) {
  nvtx3::scoped_range r("cu::SharedEvent::wait(s)");
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { wait(value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    wait(encoder.stream(), value);
    encoder.add_completed_handler([buf = buf_]() {});
  }
}

void SharedEvent::signal(uint64_t value) {
  nvtx3::scoped_range r("cu::SharedEvent::signal");
  event_signal(atomic(), value);
}

void SharedEvent::signal(cudaStream_t stream, uint64_t value) {
  event_signal_kernel<<<1, 1, 0, stream>>>(atomic(), value);
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
    encoder.add_completed_handler([buf = buf_]() {});
  }
}

bool SharedEvent::is_signaled(uint64_t value) const {
  nvtx3::scoped_range r("cu::SharedEvent::is_signaled");
  return atomic()->load() >= value;
}

uint64_t SharedEvent::value() const {
  nvtx3::scoped_range r("cu::SharedEvent::value");
  return atomic()->load();
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
  std::unique_ptr<cu::CudaEventWrapper> cuda;
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
      cuda = std::make_unique<cu::CudaEventWrapper>();
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
    return event->cuda->is_signaled();
  } else {
    return event->shared->is_signaled(value());
  }
}

} // namespace mlx::core
