// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/cuda/event.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  cu::SharedEvent event;
};

Fence::Fence(Stream s) {
  fence_ = std::shared_ptr<void>(
      new FenceImpl{0}, [](void* ptr) { delete static_cast<FenceImpl*>(ptr); });
}

void Fence::wait(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->event.wait(fence->count);
}

void Fence::update(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  fence->event.signal(s, fence->count);
}

} // namespace mlx::core
