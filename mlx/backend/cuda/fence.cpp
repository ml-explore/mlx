// Copyright © 2024 Apple Inc.

#include "mlx/fence.h"
#include "mlx/event.h"

namespace mlx::core {

struct FenceImpl {
  Event event;
  uint32_t count;
};

Fence::Fence(Stream stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{Event(stream), 0}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->event.wait(stream);
}

void Fence::update(Stream stream, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  fence->event.set_value(fence->count);
  fence->event.signal(stream);
}

} // namespace mlx::core
