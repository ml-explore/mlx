// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/event.h"
#include "mlx/backend/cpu/encoder.h"

namespace mlx::core {

namespace {

std::shared_ptr<std::string> message_from_exception(std::exception_ptr error) {
  if (!error) {
    return std::make_shared<std::string>("Unknown exception.");
  }
  try {
    std::rethrow_exception(std::move(error));
  } catch (const std::exception& e) {
    return std::make_shared<std::string>(e.what());
  } catch (...) {
    return std::make_shared<std::string>("Unknown exception.");
  }
}

} // namespace

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

void EventImpl::set_error(std::exception_ptr error, uint64_t value) {
  auto message = message_from_exception(error);
  {
    std::lock_guard<std::mutex> lk(exception_mtx_);
    if (!exception_) {
      exception_ = std::move(error);
    }
  }
  if (!std::atomic_load(&error_)) {
    std::atomic_store(&error_, std::move(message));
  }
  signal(value);
}

void EventImpl::check_error() {
  std::exception_ptr exception;
  {
    std::lock_guard<std::mutex> lk(exception_mtx_);
    exception = std::move(exception_);
    exception_ = nullptr;
  }
  if (exception) {
    std::atomic_exchange(&error_, {});
    std::rethrow_exception(exception);
  }

  auto error = std::atomic_exchange(&error_, {});
  if (error) {
    throw std::runtime_error(*error);
  }
}

std::exception_ptr EventImpl::exception() const {
  {
    std::lock_guard<std::mutex> lk(exception_mtx_);
    if (exception_) {
      return exception_;
    }
  }

  auto error = std::atomic_load(&error_);
  if (error) {
    return std::make_exception_ptr(std::runtime_error(*error));
  }
  return nullptr;
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
}

void Event::wait(Stream stream) {
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
    cpu::get_command_encoder(stream).dispatch(
        [impl = std::move(impl), value = value()]() { impl->wait(value); });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.wait_event(std::move(impl), value());
  }
}

void Event::signal(Stream stream) {
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
    cpu::get_command_encoder(stream).dispatch_unchecked(
        [impl = std::move(impl), value = value()]() { impl->signal(value); });
  } else {
    auto& encoder = metal::get_command_encoder(stream);
    encoder.signal_event(std::move(impl), value());
  }
}

bool Event::is_signaled() const {
  auto* impl = static_cast<metal::EventImpl*>(event_.get());
  impl->check_error();
  return impl->mtl_event()->signaledValue() >= value();
}

void Event::set_error(std::exception_ptr error) {
  static_cast<metal::EventImpl*>(event_.get())
      ->set_error(std::move(error), value());
}

std::exception_ptr Event::error() const {
  return static_cast<metal::EventImpl*>(event_.get())->exception();
}

} // namespace mlx::core
