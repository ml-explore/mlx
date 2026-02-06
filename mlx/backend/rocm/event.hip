// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/allocator.h"
#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/event.h"
#include "mlx/event.h"
#include "mlx/scheduler.h"

#include <map>
#include <vector>

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
  (void)hipEventSynchronize(event_);
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
            hipEventDisableTiming | hipEventBlockingSync)) {}

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

AtomicEvent::AtomicEvent() {
  buf_ = std::shared_ptr<allocator::Buffer>(
      new allocator::Buffer{allocator().malloc(sizeof(std::atomic<uint64_t>))}, 
      [](allocator::Buffer* ptr) {
        allocator().free(*ptr);
        delete ptr;
      });
  *static_cast<uint64_t*>(buf_->raw_ptr()) = 0;
}

void AtomicEvent::wait(uint64_t value) {
  auto* ac = atomic();
  uint64_t current;
  while ((current = ac->load()) < value) {
    // Spin wait
  }
}

void AtomicEvent::wait(hipStream_t stream, uint64_t value) {
  // For HIP, we use host function callback for synchronization
  (void)hipStreamSynchronize(stream);
  wait(value);
}

void AtomicEvent::wait(Stream s, uint64_t value) {
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
  atomic()->store(value);
}

void AtomicEvent::signal(hipStream_t stream, uint64_t value) {
  (void)hipStreamSynchronize(stream);
  signal(value);
}

void AtomicEvent::signal(Stream s, uint64_t value) {
  if (s.device == mlx::core::Device::cpu) {
    static HipStream stream(device(mlx::core::Device::gpu));
    scheduler::enqueue(s, [*this, value]() mutable { signal(stream, value); });
  } else {
    auto& encoder = get_command_encoder(s);
    encoder.commit();
    signal(encoder.stream(), value);
    encoder.add_completed_handler([buf = buf_]() {});
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
  if (event->hip) {
    assert(value() == 1);
    event->hip->record(s);
  } else {
    event->atomic->signal(s, value());
  }
}

bool Event::is_signaled() const {
  auto* event = static_cast<EventImpl*>(event_.get());
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
