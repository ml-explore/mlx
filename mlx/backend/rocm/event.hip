// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <map>
#include <mutex>
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

// Manage cached hipEvent_t objects. Keyed by (device, flags): a hipEvent is
// bound to the device current when it was created, and recording it on another
// device's stream is invalid (hangs the queue on a multi-GPU host). So pool and
// hand back events per device, creating new ones on the CURRENT device.
struct HipEventPool {
  static HipEventHandle create(int flags) {
    int dev = 0;
    (void)hipGetDevice(&dev);
    auto& cache = cache_for(dev, flags);
    if (cache.empty()) {
      return HipEventHandle(flags); // created on the current device
    } else {
      HipEventHandle ret = std::move(cache.back());
      cache.pop_back();
      return ret;
    }
  }

  static void release(HipEventHandle event) {
    cache_for(event.device, event.flags).push_back(std::move(event));
  }

  static std::vector<HipEventHandle>& cache_for(int device, int flags) {
    static std::map<std::pair<int, int>, std::vector<HipEventHandle>> cache;
    return cache[{device, flags}];
  }
};

} // namespace

HipEventHandle::HipEventHandle(int flags) : flags(flags) {
  CHECK_HIP_ERROR(hipEventCreateWithFlags(&handle_, flags));
  assert(handle_ != nullptr);
  (void)hipGetDevice(&device); // event is bound to the current device
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

// Pool of 8-byte SIGNAL-MEMORY counters for AtomicEvent. The counter is used two
// ways: the GPU waits on it via hipStreamWaitValue64 (cross-stream Fence) and
// signals it via hipStreamWriteValue64, AND the CPU polls it (host wait()). HIP
// REQUIRES the pointer passed to hipStreamWaitValue64/WriteValue64 to be
// allocated with the hipMallocSignalMemory flag — plain hipHostMalloc memory is
// accepted by the call (returns success) but the GPU-side wait never observes the
// value, so on a discrete GPU the stream spins forever (busy 100%, mem-ctrl 0%).
// Signal memory is host-accessible, so the CPU poll works too. Recycle the blocks
// (hipFree blocks on a discrete GPU); intentionally leaked at process exit.
struct PinnedCounterPool {
  std::mutex m;
  std::vector<void*> free_list;

  void* acquire() {
    {
      std::lock_guard<std::mutex> lk(m);
      if (!free_list.empty()) {
        void* p = free_list.back();
        free_list.pop_back();
        return p;
      }
    }
    void* p = nullptr;
    CHECK_HIP_ERROR(hipHostMalloc(
        &p, sizeof(std::atomic<uint64_t>),
        hipHostMallocMapped | hipHostMallocCoherent));
    return p;
  }

  void release(void* p) {
    std::lock_guard<std::mutex> lk(m);
    free_list.push_back(p);
  }
};

PinnedCounterPool& counter_pool() {
  static PinnedCounterPool* pool = new PinnedCounterPool;
  return *pool;
}

} // namespace

AtomicEvent::AtomicEvent() {
  // Completion counter in pinned, device-mapped host memory: the GPU signals it
  // (hipStreamWriteValue64) and the CPU polls it (wait()). On a discrete GPU over
  // a non-coherent link (TB5 eGPU) device memory is NOT CPU-visible, so the
  // counter must live in host memory or the host poll spins forever. Drawn from a
  // recycling pool — see PinnedCounterPool (hipHostFree blocks on a discrete GPU).
  void* p = counter_pool().acquire();
  atomic_ = static_cast<std::atomic<uint64_t>*>(p);
  atomic_->store(0, std::memory_order_release);
  mem_ = std::shared_ptr<void>(p, [](void* q) { counter_pool().release(q); });
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
      // Bind the stream's device current before creating the hipEvent — it is
      // bound to whatever device is current at creation, and recording it on a
      // different device's stream hangs the queue on a multi-GPU host.
      (void)rocm::get_command_encoder(s);
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
