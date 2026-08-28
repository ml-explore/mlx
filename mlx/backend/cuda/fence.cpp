// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  Stream producer;
  Event gpu_event;
  Event cpu_event;

  FenceImpl(uint32_t count, Stream s)
      : count(count), producer(s), cpu_event(s) {
    if (s.device == Device::gpu) {
      gpu_event = Event(s);
      // A value of one selects a native CUDA event.
      gpu_event.set_value(1);
    }
  }
};

Fence::Fence(Stream s) {
  fence_ = std::make_shared<FenceImpl>(0, s);
  auto& f = cast<FenceImpl>();
  // Ensure that we use AtomicEvent, it is the only event that can order a CPU
  // stream against the GPU.
  f.cpu_event.cast<cu::EventImpl>().ensure_created(s, 2);
}

void Fence::wait(Stream s, const array&) {
  auto& f = cast<FenceImpl>();
  if (f.count == 0) {
    return;
  }
  if (f.producer.device == Device::gpu && s.device == Device::gpu) {
    f.gpu_event.wait(s);
  } else {
    // AtomicEvent can not reliably notify a GPU stream, so a dependency that
    // involves the CPU keeps the synchronous wait.
    f.cpu_event.wait();
  }
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
  if (s.device == Device::gpu) {
    f.gpu_event.signal(s);
  }
  // The counted event stays current, so a CPU consumer is always ordered
  // against every update.
  f.cpu_event.set_value(f.count);
  f.cpu_event.signal(s);
}

} // namespace mlx::core
