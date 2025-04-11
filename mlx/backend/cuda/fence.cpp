// Copyright © 2024 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/cuda/event.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  mxcuda::SharedEvent event;
};

Fence::Fence(Stream stream) {
  fence_ = std::shared_ptr<void>(
      new FenceImpl{0}, [](void* ptr) { delete static_cast<FenceImpl*>(ptr); });
}

void Fence::wait(Stream stream, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->event.wait(stream, fence->count);
}

void Fence::update(Stream stream, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  fence->event.signal(stream, fence->count);
}

} // namespace mlx::core
