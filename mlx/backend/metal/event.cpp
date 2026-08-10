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
  check_error();
  mtl_event_->waitUntilSignaledValue(value, -1); // never times out
  check_error();
}

void EventImpl::signal(uint64_t value) {
  mtl_event_->setSignaledValue(value);
}

void EventImpl::set_error(std::shared_ptr<std::string> error) {
  std::atomic_store(&error_, std::move(error));
}

void EventImpl::check_error() {
  auto error = std::atomic_exchange(&error_, {});
  if (error) {
    throw std::runtime_error(*error);
  }
}

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Event implementations
///////////////////////////////////////////////////////////////////////////////

Event::Event(Stream stream) : stream_(stream) {
  event_ = std::make_shared<metal::EventImpl>(metal::device(stream.device));
}

void Event::wait() {
  try {
    static_cast<metal::EventImpl*>(event_.get())->wait(value());
  } catch (...) {
    // Event already reported the failure; drop the sticky copy.
    metal::clear_stream_error(stream_);
    throw;
  }
  // Late attribution: a prior command buffer on this stream may have failed
  // with no signal event, leaving only the sticky stream error.
  metal::check_stream_error(stream_);
}

void Event::wait(Stream stream) {
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
    scheduler::enqueue(
        stream,
        [impl = std::move(impl), value = value(), event_stream = stream_]() {
          try {
            impl->wait(value);
          } catch (...) {
            metal::clear_stream_error(event_stream);
            throw;
          }
          metal::check_stream_error(event_stream);
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
