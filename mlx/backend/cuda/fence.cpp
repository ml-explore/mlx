// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  cu::AtomicEvent event;
};

Fence::Fence(Stream s) {
  fence_ = std::shared_ptr<void>(
      new FenceImpl{0, cu::device(s.device)},
      [](void* ptr) { delete static_cast<FenceImpl*>(ptr); });
}

void Fence::wait(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->event.wait(fence->count);
}

void Fence::update(Stream s, const array& a, bool cross_device) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  if (cross_device) {
    // Move to managed memory if there is a device switch
    auto& cbuf =
        *static_cast<cu::CudaBuffer*>(const_cast<array&>(a).buffer().ptr());
    if (cbuf.device != -1) {
      auto& encoder = cu::get_command_encoder(s);
      encoder.commit();
      cu::allocator().move_to_unified_memory(cbuf, encoder.stream());
    }
  }
  fence->count++;
  fence->event.signal(s, fence->count);
}

} // namespace mlx::core
