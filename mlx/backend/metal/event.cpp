// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/event.h"
#include "mlx/backend/gpu/eval.h"
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
  command_buffer_.reset();
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

void EventImpl::set_command_buffer(MTL::CommandBuffer* command_buffer) {
  command_buffer_ = NS::RetainPtr(command_buffer);
}

double EventImpl::gpu_end_time() {
  if (!command_buffer_) {
    throw std::runtime_error(
        "[Event::elapsed_time] Event does not have a command buffer.");
  }
  command_buffer_->waitUntilCompleted();
  auto end_time = command_buffer_->GPUEndTime();
  if (end_time == 0.0) {
    throw std::runtime_error(
        "[Event::elapsed_time] Metal GPU end time is unavailable.");
  }
  return end_time;
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

Event::Event(Stream stream, bool enable_timing)
    : stream_(stream), enable_timing_(enable_timing) {
  if (stream.device != Device::gpu) {
    throw std::invalid_argument(
        "[Event::Event] Events are only supported on GPU.");
  }
  event_ = std::make_shared<metal::EventImpl>(metal::device(stream.device));
}

void Event::record(Stream stream) {
  if (stream.device != Device::gpu) {
    throw std::invalid_argument(
        "[Event::record] Events can only be recorded on GPU streams.");
  }
  if (stream.device != stream_.device) {
    throw std::invalid_argument(
        "[Event::record] The event and stream must use the same device.");
  }
  stream_ = stream;
  recorded_ = true;
  value_++;

  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  auto& encoder = metal::get_command_encoder(stream);
  encoder.record_event(std::move(impl), value(), enable_timing_);
  gpu::finalize(stream);
}

double Event::elapsed_time(const Event& end_event) const {
  if (!enable_timing_ || !end_event.enable_timing_) {
    throw std::runtime_error(
        "[Event::elapsed_time] Both events must have timing enabled.");
  }
  if (!recorded_ || !end_event.recorded_) {
    throw std::runtime_error(
        "[Event::elapsed_time] Both events must be recorded first.");
  }
  if (stream_.device != end_event.stream_.device) {
    throw std::invalid_argument(
        "[Event::elapsed_time] Events must use the same device.");
  }

  auto start_impl = std::static_pointer_cast<metal::EventImpl>(event_);
  auto end_impl = std::static_pointer_cast<metal::EventImpl>(end_event.event_);
  start_impl->wait(value());
  end_impl->wait(end_event.value());
  auto start = start_impl->gpu_end_time();
  auto end = end_impl->gpu_end_time();
  return (end - start) * 1e3;
}

void Event::wait() {
  if (value() == 0) {
    return;
  }
  static_cast<metal::EventImpl*>(event_.get())->wait(value());
}

void Event::wait(Stream stream) {
  if (value() == 0) {
    return;
  }
  auto impl = std::static_pointer_cast<metal::EventImpl>(event_);
  if (stream.device == Device::cpu) {
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
