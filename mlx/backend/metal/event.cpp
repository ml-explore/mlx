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
}

void EventImpl::signal(uint64_t value) {
  mtl_event_->setSignaledValue(value);
}

void EventImpl::set_error(std::shared_ptr<std::string> error) {
  std::atomic_store(&error_, std::move(error));
}

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

Event::Event(Stream stream) : stream_(stream) {
  event_ = std::make_shared<metal::EventImpl>(metal::device(stream.device));
}

void Event::wait() {
  static_cast<metal::EventImpl*>(event_.get())->wait(value());
  // The event being signaled says nothing about whether the work succeeded:
  // Metal signals on the GPU timeline, ahead of the completion handler that
  // records the failure, and a command buffer can die with no event attached
  // at all. The stream error covers both.
  metal::check_stream_error(stream_);
}

void Event::wait(Stream stream) {
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
    // Deliberately does not report errors. This runs on a scheduler worker
    // with no handler above it, so throwing here would terminate the process
    // instead of reaching the user. The failure stays parked on the signaling
    // stream and surfaces at the next synchronization point.
    scheduler::enqueue(stream, [impl = std::move(impl), value = value()]() {
      impl->wait(value);
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.wait_event(std::move(impl), value());
  }
}

void Event::signal(Stream stream) {
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [impl = std::move(impl), value = value()]() {
      impl->signal(value);
    });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.signal_event(std::move(impl), value());
  }
}

bool Event::is_signaled() const {
  auto* mtl_event = static_cast<metal::EventImpl*>(event_.get())->mtl_event();
  return mtl_event->signaledValue() >= value();
}

} // namespace mlx::core
