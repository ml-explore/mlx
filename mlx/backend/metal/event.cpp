// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal_impl.h"

namespace mlx::core {

void encode_wait(Event e) {
  auto& d = metal::device(e.stream().device);
  d.end_encoding(e.stream().index);
  auto command_buffer = d.get_command_buffer(e.stream().index);
  command_buffer->encodeWait(
      static_cast<MTL::Event*>(e.raw_event().get()), e.value());
  command_buffer->addCompletedHandler(
      [e = std::move(e)](MTL::CommandBuffer* cbuf) {});
}

void encode_signal(Event e) {
  auto& d = metal::device(e.stream().device);
  d.end_encoding(e.stream().index);
  auto command_buffer = d.get_command_buffer(e.stream().index);
  command_buffer->encodeSignalEvent(
      static_cast<MTL::Event*>(e.raw_event().get()), e.value());
  command_buffer->addCompletedHandler(
      [e = std::move(e)](MTL::CommandBuffer* cbuf) {});
}

Event::Event(const Stream& stream) : stream_(stream) {
  auto dtor = [](void* ptr) {
    auto p = metal::new_scoped_memory_pool();
    static_cast<MTL::SharedEvent*>(ptr)->release();
  };
  auto p = metal::new_scoped_memory_pool();
  event_ = std::shared_ptr<void>(
      metal::device(stream.device).mtl_device()->newSharedEvent(), dtor);
}

void Event::wait() {
  if (!static_cast<MTL::SharedEvent*>(raw_event().get())
           ->waitUntilSignaledValue(value(), -1)) {
    throw std::runtime_error("[Event::wait] Timed out");
  }
}

void Event::signal() {
  static_cast<MTL::SharedEvent*>(raw_event().get())->setSignaledValue(value());
}

bool Event::is_signaled() const {
  return static_cast<MTL::SharedEvent*>(raw_event().get())->signaledValue() >=
      value();
}

} // namespace mlx::core
