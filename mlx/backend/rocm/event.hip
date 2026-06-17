// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <map>
#include <thread>
#include <vector>
#include <immintrin.h>

#include <hip/hip_runtime.h>

namespace mlx::core {

namespace rocm {

///////////////////////////////////////////////////////////////////////////////
// HipEvent implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

// Manage cached hipEvent_t objects.
struct HipEventPool {
  static HipEventHandle create(int flags) {
    auto& cache = cache_for(flags);
    if (cache.empty()) {
      return HipEventHandle(flags);
    } else {
      HipEventHandle ret = std::move(cache.back());
      cache.pop_back();
      return ret;
    }
  }

  static void release(HipEventHandle event) {
    cache_for(event.flags).push_back(std::move(event));
  }

  static std::vector<HipEventHandle>& cache_for(int flags) {
    static std::map<int, std::vector<HipEventHandle>> cache;
    return cache[flags];
  }
};

} // namespace

HipEventHandle::HipEventHandle(int flags) : flags(flags) {
  CHECK_HIP_ERROR(hipEventCreateWithFlags(&handle_, flags));
  assert(handle_ != nullptr);
}

HipEvent::HipEvent(int flags) : event_(HipEventPool::create(flags)) {}

HipEvent::~HipEvent() {
  HipEventPool::release(std::move(event_));
}

void HipEvent::wait() {
  // Spin-wait with hipEventQuery instead of hipEventSynchronize.
  // On iGPU, the blocking wait in hipEventSynchronize causes CPU-GPU
  // contention since they share compute resources. Polling is cheaper.
  // Use progressive backoff to reduce hipEventQuery call overhead.
  for (int spins = 0; hipEventQuery(event_) != hipSuccess; spins++) {
    if (spins < 100) {
      // Tight spin for fast completions
    } else if (spins < 1000) {
      _mm_pause(); // x86 pause hint (reduces power, avoids pipeline stall)
    } else {
      std::this_thread::yield();
    }
  }
}

void HipEvent::wait(hipStream_t stream) {
  (void)hipStreamWaitEvent(stream, event_, 0);
}

void HipEvent::record(hipStream_t stream) {
  (void)hipEventRecord(event_, stream);
}

bool HipEvent::completed() const {
  return hipEventQuery(event_) == hipSuccess;
}

// Wraps HipEvent with a few features:
// 1. The class can be copied.
// 2. Make wait/record work with CPU streams.
// 3. Add checks for waiting on un-recorded event.
class CopyableHipEvent {
 public:
  CopyableHipEvent()
      : event_(std::make_shared<HipEvent>(
            hipEventDisableTiming)) {}
            // Note: hipEventBlockingSync removed — on iGPU the blocking wait
            // contends with GPU for CPU resources. Polling is cheaper.

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
      auto& encoder = rocm::get_command_encoder(s);
      encoder.commit();
      event_->wait(encoder.stream());
    }
  }

  void record(Stream s) {
    if (s.device == mlx::core::Device::cpu) {
      throw std::runtime_error("HipEvent can not wait on CPU stream.");
    } else {
      auto& encoder = rocm::get_command_encoder(s);
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
          "Should not wait on a HipEvent before recording.");
    }
  }

  std::shared_ptr<HipEvent> event_;
  bool recorded_{false};
};

///////////////////////////////////////////////////////////////////////////////
// AtomicEvent implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

void signal_atomic_callback(void* data) {
  auto* pair = static_cast<std::pair<std::atomic<uint64_t>*, uint64_t>*>(data);
  pair->first->store(pair->second);
  delete pair;
}

} // namespace

AtomicEvent::AtomicEvent() {
  // Pinned, device-mapped host memory for the completion counter. The GPU
  // signals it (hipStreamWriteValue64) and the CPU polls it (wait()); on a
  // discrete GPU over a non-coherent link (TB5 eGPU) device memory is not
  // visible to the CPU, so the counter MUST live in host memory or the host
  // poll spins forever. Mapped+coherent so the same pointer is valid on the
  // device (HIP unified addressing) for the stream write/wait-value ops.
  void* p = nullptr;
  CHECK_HIP_ERROR(hipHostMalloc(
      &p, sizeof(std::atomic<uint64_t>),
      hipHostMallocMapped | hipHostMallocCoherent));
  atomic_ = static_cast<std::atomic<uint64_t>*>(p);
  atomic_->store(0, std::memory_order_release);
  mem_ = std::shared_ptr<void>(p, [](void* q) { (void)hipHostFree(q); });
}

void AtomicEvent::wait(uint64_t value) {
  auto* ac = atomic();
  while (ac->load(std::memory_order_acquire) < value) {
    std::this_thread::yield();
  }
}

void AtomicEvent::wait(hipStream_t stream, uint64_t value) {
  // Use hipStreamWaitValue64 if possible to make the GPU wait for the atomic directly.
  // This avoids blocking the host thread and is much more efficient.
  // flags = hipStreamWaitValueGte (Greater than or equal)
  hipError_t err = hipStreamWaitValue64(stream, atomic(), value, hipStreamWaitValueGte, 0xFFFFFFFFFFFFFFFFULL);
  if (err != hipSuccess) {
    // Fallback to synchronous wait if hipStreamWaitValue64 is not supported or fails.
    // hipStreamSynchronize should be blocking if flags are set correctly.
    CHECK_HIP_ERROR(hipStreamSynchronize(stream));
    wait(value);
  }
}

void AtomicEvent::wait(Stream s, uint64_t value) {
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { wait(value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    wait(encoder.stream(), value);
    // Keep the buffer alive until the wait is finished
    encoder.add_completed_handler([buf = mem_]() {});
  }
}

void AtomicEvent::signal(uint64_t value) {
  atomic()->store(value, std::memory_order_release);
}

void AtomicEvent::signal(hipStream_t stream, uint64_t value) {
  // Use hipStreamWriteValue64 if possible to signal the atomic directly from the GPU stream.
  // This is much more efficient than using a host callback.
  // We don't use flags or mask for now.
  hipError_t err = hipStreamWriteValue64(stream, atomic(), value, 0);
  if (err != hipSuccess) {
    // Fallback to host callback if hipStreamWriteValue64 is not supported or fails.
    auto* data = new std::pair<std::atomic<uint64_t>*, uint64_t>(atomic(), value);
    CHECK_HIP_ERROR(hipLaunchHostFunc(stream, signal_atomic_callback, data));
  }
}

void AtomicEvent::signal(Stream s, uint64_t value) {
  if (s.device == mlx::core::Device::cpu) {
    scheduler::enqueue(s, [*this, value]() mutable { signal(value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    signal(encoder.stream(), value);
    // Keep the buffer alive until it's signaled
    encoder.add_completed_handler([buf = mem_]() {});
  }
}

bool AtomicEvent::is_signaled(uint64_t value) const {
  return atomic()->load() >= value;
}

uint64_t AtomicEvent::value() const {
  return atomic()->load();
}

} // namespace rocm

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

namespace {

struct EventImpl {
  std::unique_ptr<rocm::CopyableHipEvent> hip;
  std::unique_ptr<rocm::AtomicEvent> atomic;
  // Set when the event is "signaled" purely on the host because it was created
  // during HIP graph capture: recording a real completion event onto the
  // captured stream would bake it into the graph (never fires under capture),
  // so we skip the device record and treat the event as already satisfied. The
  // recorded compute kernels still execute on replay; completion is handled by
  // an explicit stream sync after hipGraphLaunch.
  bool host_signaled{false};

  bool is_created() const {
    return hip || atomic;
  }

  void ensure_created(Stream s, uint64_t signal_value) {
    if (is_created()) {
      return;
    }
    if (s.device == mlx::core::Device::cpu || signal_value > 1) {
      atomic = std::make_unique<rocm::AtomicEvent>();
    } else {
      hip = std::make_unique<rocm::CopyableHipEvent>();
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
  // Capture-created event: nothing to wait for on the host (see EventImpl).
  if (event->host_signaled) {
    return;
  }
  assert(event->is_created());
  if (event->hip) {
    assert(value() == 1);
    event->hip->wait();
  } else {
    event->atomic->wait(value());
  }
}

void Event::wait(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  if (event->host_signaled) {
    return;
  }
  assert(event->is_created());
  if (event->hip) {
    assert(value() == 1);
    event->hip->wait(s);
  } else {
    event->atomic->wait(s, value());
  }
}

void Event::signal(Stream s) {
  auto* event = static_cast<EventImpl*>(event_.get());
  event->ensure_created(s, value());
  // During graph capture, do NOT record a completion event onto the captured
  // stream — it would become a graph node that never fires under capture and
  // would deadlock eval()'s wait(). Mark it satisfied on the host instead; the
  // compute kernels are recorded and run on replay, after which the caller
  // performs an explicit stream sync.
  if (!(s.device == mlx::core::Device::cpu) &&
      rocm::get_command_encoder(s).capturing()) {
    event->host_signaled = true;
    return;
  }
  if (event->hip) {
    assert(value() == 1);
    event->hip->record(s);
  } else {
    event->atomic->signal(s, value());
  }
}

bool Event::is_signaled() const {
  auto* event = static_cast<EventImpl*>(event_.get());
  if (event->host_signaled) {
    return true;
  }
  if (!event->is_created()) {
    return false;
  }
  if (event->hip) {
    assert(value() == 1);
    return event->hip->is_signaled();
  } else {
    return event->atomic->is_signaled(value());
  }
}

} // namespace mlx::core
