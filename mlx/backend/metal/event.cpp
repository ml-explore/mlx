// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core {

Event::Event(const Stream& stream) : stream_(stream) {
  auto dtor = [](void* ptr) { static_cast<MTL::SharedEvent*>(ptr)->release(); };
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

} // namespace mlx::core
