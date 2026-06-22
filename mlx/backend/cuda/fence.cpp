// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  Event event;

  FenceImpl(uint32_t count, Stream s) : count(count), event(s) {}
};

Fence::Fence(Stream s) {
  fence_ = std::make_shared<FenceImpl>(0, s);
  // Ensure that we use AtomicEvent.
  cast<FenceImpl>().event.cast<cu::EventImpl>().ensure_created(s, 2);
}

void Fence::wait(Stream s, const array&) {
  cast<FenceImpl>().event.wait();
}

void Fence::update(Stream s, const array& a, bool cross_device) {
  auto& f = cast<FenceImpl>();
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
  f.count++;
  f.event.set_value(f.count);
  f.event.signal(s);
}

} // namespace mlx::core
