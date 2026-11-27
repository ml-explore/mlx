// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/event.h"
#include "mlx/scheduler.h"

namespace mlx::core {

///////////////////////////////////////////////////////////////////////////////
// EventImpl implementations
///////////////////////////////////////////////////////////////////////////////

namespace metal {

EventImpl::EventImpl(Device& d) {
  auto p = new_scoped_memory_pool();
  mtl_event_ = NS::TransferPtr(d.mtl_device()->newSharedEvent());
  if (!mtl_event_) {
    throw std::runtime_error(
        "[Event::Event] Failed to create Metal shared event.");
  }
}

EventImpl::~EventImpl() {
  auto p = new_scoped_memory_pool();
  mtl_event_.reset();
}

void EventImpl::wait(uint64_t value) {
  mtl_event_->waitUntilSignaledValue(value, -1); // never times out
  drop_refs(value);
}

void EventImpl::drop_refs(uint64_t value) {
  // The GPU ran all queue commands before the signal, also in earlier command
  // buffers. Eval relies on the same order.
  std::shared_ptr<DataRefs> refs;
  {
    std::lock_guard lk(refs_mtx_);
    if (value >= refs_value_) {
      refs = refs_.lock();
    }
  }
  if (refs) {
    refs->drop_shared();
  }
}

void EventImpl::set_refs(uint64_t value, std::weak_ptr<DataRefs> refs) {
  std::lock_guard lk(refs_mtx_);
  if (value > refs_value_) {
    refs_value_ = value;
    refs_ = std::move(refs);
  } else {
    refs_.reset();
  }
}

void EventImpl::signal(uint64_t value) {
  mtl_event_->setSignaledValue(value);
}

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

Event::Event(Stream stream) : stream_(stream) {
  event_ = std::make_shared<metal::EventImpl>(metal::device(stream.device));
}

void Event::wait() {
  check_error();
  cast<metal::EventImpl>().wait(value());
  check_error();
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::wait_event(stream, *this, [value = value()](Event& self) {
      self.cast<metal::EventImpl>().wait(value);
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.wait_event(*this, value());
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::signal_event(stream, *this, [value = value()](Event& self) {
      self.cast<metal::EventImpl>().signal(value);
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.signal_event(*this, value());
  }
}

bool Event::is_signaled() const {
  auto& impl = cast<metal::EventImpl>();
  if (impl.mtl_event()->signaledValue() < value()) {
    return false;
  }
  impl.drop_refs(value());
  return true;
}

std::atomic<Error*>& Event::error() {
  return cast<metal::EventImpl>().error();
}

} // namespace mlx::core
