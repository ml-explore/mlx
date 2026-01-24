// Copyright © 2024 Apple Inc.

#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <map>
#include <vector>

#include <nvtx3/nvtx3.hpp>

namespace mlx::core {

namespace cu {

///////////////////////////////////////////////////////////////////////////////
// CudaEvent implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

// Manage cached cudaEvent_t objects.
class CudaEventPool {
 public:
  CudaEventHandle create(Device& d, int flags) {
    if (!on_creation_thread()) {
      return CudaEventHandle(d, flags);
    }
    auto& cache = cache_for(d, flags);
    if (cache.empty()) {
      return CudaEventHandle(d, flags);
    } else {
      CudaEventHandle ret = std::move(cache.back());
      cache.pop_back();
      return ret;
    }
  }

  void release(CudaEventHandle event) {
    if (!on_creation_thread()) {
      // Event will be destroyed directly instead of getting moved to cache.
      return;
    }
    cache_for(event.device, event.flags).push_back(std::move(event));
  }

 private:
  std::vector<CudaEventHandle>& cache_for(Device& d, int flags) {
    return cache_[d.cuda_device()][flags];
  }

  bool on_creation_thread() {
    return std::this_thread::get_id() == thread_id_;
  }

  // The CudaEvent may be created and destroyed on different threads (for
  // example when waiting on GPU work in CPU stream), we don't want to make
  // the cache thread-safe as it adds overhead, so we just skip cache when
  // using events in worker threads.
  std::thread::id thread_id_{std::this_thread::get_id()};

  // {device: {flags: [events]}}
  std::map<int, std::map<int, std::vector<CudaEventHandle>>> cache_;
};

CudaEventPool& cuda_event_pool() {
  static CudaEventPool pool;
  return pool;
}

} // namespace

CudaEventHandle::CudaEventHandle(Device& d, int flags)
    : device(d), flags(flags) {
  device.make_current();
  CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&handle_, flags));
  assert(handle_ != nullptr);
}

CudaEvent::CudaEvent(Device& d, int flags)
    : event_(cuda_event_pool().create(d, flags)) {}

CudaEvent::~CudaEvent() {
  cuda_event_pool().release(std::move(event_));
}

void CudaEvent::wait() {
  nvtx3::scoped_range r("cu::CudaEvent::wait");
  event_.device.make_current();
  cudaEventSynchronize(event_);
}

void CudaEvent::wait(cudaStream_t stream) {
  event_.device.make_current();
  cudaStreamWaitEvent(stream, event_);
}

void CudaEvent::record(cudaStream_t stream) {
  event_.device.make_current();
  cudaEventRecord(event_, stream);
}

bool CudaEvent::completed() const {
  // Note: cudaEventQuery can be safely called from any device.
  return cudaEventQuery(event_) == cudaSuccess;
}

// static
void CudaEvent::init_pool() {
  cuda_event_pool();
}

// Wraps CudaEvent with a few features:
// 1. The class can be copied.
// 2. Make wait/record work with CPU streams.
// 3. Add checks for waiting on un-recorded event.
class CopyableCudaEvent {
 public:
  explicit CopyableCudaEvent(Device& d)
      : event_(std::make_shared<CudaEvent>(
            d,
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
// AtomicEvent implementations
///////////////////////////////////////////////////////////////////////////////

__host__ __device__ void event_wait(AtomicEvent::Atomic* ac, uint64_t value) {
  uint64_t current;
  while ((current = ac->load()) < value) {
    ac->wait(current);
  }
}

__host__ __device__ void event_signal(AtomicEvent::Atomic* ac, uint64_t value) {
  ac->store(value);
  ac->notify_all();
}

__global__ void event_wait_kernel(AtomicEvent::Atomic* ac, uint64_t value) {
  event_wait(ac, value);
}

__global__ void event_signal_kernel(AtomicEvent::Atomic* ac, uint64_t value) {
  event_signal(ac, value);
}

AtomicEvent::AtomicEvent() {
  buf_ = std::shared_ptr<Buffer>(
      new Buffer{allocator().malloc(sizeof(Atomic))}, [](Buffer* ptr) {
        allocator().free(*ptr);
        delete ptr;
      });
  *static_cast<uint64_t*>(buf_->raw_ptr()) = 0;
}

void AtomicEvent::wait(uint64_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::wait");
  event_wait(atomic(), value);
}

void AtomicEvent::wait(cudaStream_t stream, uint64_t value) {
  event_wait_kernel<<<1, 1, 0, stream>>>(atomic(), value);
}

void AtomicEvent::wait(Stream s, uint64_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::wait(s)");
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { wait(value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    wait(encoder.stream(), value);
    encoder.add_completed_handler([buf = buf_]() {});
  }
}

void AtomicEvent::signal(uint64_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::signal");
  event_signal(atomic(), value);
}

void AtomicEvent::signal(cudaStream_t stream, uint64_t value) {
  event_signal_kernel<<<1, 1, 0, stream>>>(atomic(), value);
}

void AtomicEvent::signal(Stream s, uint64_t value) {
  nvtx3::scoped_range r("cu::AtomicEvent::signal(s)");
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

bool AtomicEvent::is_signaled(uint64_t value) const {
  nvtx3::scoped_range r("cu::AtomicEvent::is_signaled");
  return atomic()->load() >= value;
}

uint64_t AtomicEvent::value() const {
  nvtx3::scoped_range r("cu::AtomicEvent::value");
  return atomic()->load();
}

} // namespace cu

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

struct EventImpl {
  // CudaEvent is preferred when possible because it is fast, however we have
  // to fallback to AtomicEvent in following cases:
  // 1. the event is used to wait/signal a cpu stream;
  // 2. signal value other than 1 has been specified.
  std::unique_ptr<cu::CopyableCudaEvent> cuda;
  std::unique_ptr<cu::AtomicEvent> atomic;

  bool is_created() const {
    return cuda || atomic;
  }

  void ensure_created(Stream s, uint64_t signal_value) {
    if (is_created()) {
      return;
    }
    if (s.device == mlx::core::Device::cpu || signal_value > 1) {
      nvtx3::mark("Using slow AtomicEvent");
      atomic = std::make_unique<cu::AtomicEvent>();
    } else {
      cuda = std::make_unique<cu::CopyableCudaEvent>(cu::device(s.device));
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
    event->atomic->wait(value());
  }
  CHECK_CUDA_ERROR(cudaPeekAtLastError());
}

void Event::wait(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  assert(event->is_created());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->wait(s);
  } else {
    event->atomic->wait(s, value());
  }
}

void Event::signal(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  event->ensure_created(s, value());
  if (event->cuda) {
    assert(value() == 1);
    event->cuda->record(s);
  } else {
    event->atomic->signal(s, value());
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
    return event->atomic->is_signaled(value());
  }
}

} // namespace mlx::core
