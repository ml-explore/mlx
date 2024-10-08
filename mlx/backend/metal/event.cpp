// Copyright © 2024 Apple Inc.

#include "mlx/event.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal_impl.h"

namespace mlx::core {

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
