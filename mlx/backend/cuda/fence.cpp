// Copyright © 2025 Apple Inc.

#include "mlx/fence.h"
#include "mlx/backend/cpu/encoder.h"
#include "mlx/backend/cuda/allocator.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"

#include <exception>
#include <mutex>

namespace mlx::core {

struct FenceImpl {
  uint32_t count;
  cu::AtomicEvent event;
  std::exception_ptr error;
  std::mutex mtx;
};

Fence::Fence(Stream s) {
  fence_ = std::shared_ptr<void>(
      new FenceImpl{0, cu::device(s.device), nullptr, {}},
      [](void* ptr) { delete static_cast<FenceImpl*>(ptr); });
}

void Fence::wait(Stream s, const array&) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  auto count = fence->count;
  if (s.device == Device::cpu) {
    cpu::get_command_encoder(s).dispatch([fence_ = fence_, count]() mutable {
      auto* fence = static_cast<FenceImpl*>(fence_.get());
      fence->event.wait(count);
      std::exception_ptr error;
      {
        std::lock_guard<std::mutex> lk(fence->mtx);
        error = fence->error;
      }
      if (error) {
        std::rethrow_exception(error);
      }
    });
  } else {
    fence->event.wait(s, count);
  }
}

void Fence::update(Stream s, const array& a, bool cross_device) {
  auto* fence = static_cast<FenceImpl*>(fence_.get());
  fence->count++;
  auto count = fence->count;
  if (s.device == Device::cpu) {
    cpu::get_command_encoder(s).dispatch_unchecked(
        [event = a.event(), fence_ = fence_, count]() mutable {
          auto* fence = static_cast<FenceImpl*>(fence_.get());
          auto error = event.valid() ? event.error() : nullptr;
          if (error) {
            std::lock_guard<std::mutex> lk(fence->mtx);
            if (!fence->error) {
              fence->error = std::move(error);
            }
          }
          fence->event.signal(count);
        });
    return;
  }
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
  fence->event.signal(s, count);
}

} // namespace mlx::core
