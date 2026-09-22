// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"

#include <vector>

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  std::vector<Event> gpu_events;
  Event cpu_event;

  FenceImpl(uint32_t count, Stream s) : count(count), cpu_event(s) {
    // Ensure that we use AtomicEvent, it is the only event that can order a CPU
    // stream against the GPU.
    cpu_event.cast<cu::EventImpl>().ensure_created(s, 2);
  }
};

Fence::Fence(Stream s) {
  fence_ = std::make_shared<FenceImpl>(0, s);
}

void Fence::wait(Stream s, const array&, uint32_t value) {
  auto& f = cast<FenceImpl>();
  if (value == 0) {
    return;
  }
  if (!f.gpu_events.empty() && s.device == Device::gpu) {
    f.gpu_events.at(value - 1).wait(s);
  } else {
    // AtomicEvent can not reliably notify a GPU stream, so a dependency that
    // involves the CPU keeps the synchronous wait.
    auto event = f.cpu_event;
    event.set_value(value);
    event.wait();
  }
}

uint32_t Fence::update(Stream s, const array& a, bool cross_device) {
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
  if (s.device == Device::gpu) {
    // Keep each recording so a consumer can wait for an earlier update.
    auto& event = f.gpu_events.emplace_back(s);
    event.set_value(1);
    event.signal(s);
  }
  f.cpu_event.set_value(f.count);
  f.cpu_event.signal(s);
  return f.count;
}

} // namespace mlx::core
