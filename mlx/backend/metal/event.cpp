// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/metal/device.h"
#include "mlx/scheduler.h"

namespace mlx::core {

Event::Event(Stream stream) : stream_(stream) {
  auto dtor = [](void* ptr) {
    auto p = metal::new_scoped_memory_pool();
    static_cast<MTL::SharedEvent*>(ptr)->release();
  };
  auto p = metal::new_scoped_memory_pool();
  event_ = std::shared_ptr<void>(
      metal::device(Device::gpu).mtl_device()->newSharedEvent(), dtor);
}

void Event::wait() {
  if (!static_cast<MTL::SharedEvent*>(event_.get())
           ->waitUntilSignaledValue(value(), -1)) {
    throw std::runtime_error("[Event::wait] Timed out");
  }
}

void Event::wait(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable { wait(); });
  } else {
    auto& d = metal::device(stream.device);
    d.end_encoding(stream.index);
    auto command_buffer = d.get_command_buffer(stream.index);
    command_buffer->encodeWait(static_cast<MTL::Event*>(event_.get()), value());
    command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
  }
}

void Event::signal(Stream stream) {
  if (stream.device == Device::cpu) {
    scheduler::enqueue(stream, [*this]() mutable {
      static_cast<MTL::SharedEvent*>(event_.get())->setSignaledValue(value());
    });
  } else {
    auto& d = metal::device(stream.device);
    d.end_encoding(stream.index);
    auto command_buffer = d.get_command_buffer(stream.index);
    command_buffer->encodeSignalEvent(
        static_cast<MTL::Event*>(event_.get()), value());
    command_buffer->addCompletedHandler([*this](MTL::CommandBuffer*) {});
  }
}

bool Event::is_signaled() const {
  return static_cast<MTL::SharedEvent*>(event_.get())->signaledValue() >=
      value();
}

} // namespace mlx::core
