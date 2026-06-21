// Copyright © 2024 Apple Inc.

#include <condition_variable>
#include <exception>
#include <mutex>

#include "mlx/backend/cpu/encoder.h"
#include "mlx/fence.h"

namespace mlx::core {

struct FenceImpl {
  uint32_t count{0};
  uint32_t value{0};
  std::exception_ptr error;
  std::mutex mtx;
  std::condition_variable cv;
};

Fence::Fence(Stream) {
  auto dtor = [](void* ptr) { delete static_cast<FenceImpl*>(ptr); };
  fence_ = std::shared_ptr<void>(new FenceImpl{}, dtor);
}

void Fence::wait(Stream stream, const array&) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  if (stream.device == Device::cpu) {
    cpu::get_command_encoder(stream).dispatch(
        [count = f.count, fence_ = fence_]() mutable {
          auto& f = *static_cast<FenceImpl*>(fence_.get());
          std::unique_lock<std::mutex> lk(f.mtx);
          f.cv.wait(lk, [&f, count] { return f.value >= count || f.error; });
          if (f.error) {
            std::rethrow_exception(f.error);
          }
        });
  } else {
    throw std::runtime_error("[Fence::wait] Invalid stream.");
  }
}

void Fence::update(Stream stream, const array& x, bool) {
  auto& f = *static_cast<FenceImpl*>(fence_.get());
  f.count++;
  if (stream.device == Device::cpu) {
    cpu::get_command_encoder(stream).dispatch_unchecked(
        [count = f.count, event = x.event(), fence_ = fence_]() mutable {
          auto& f = *static_cast<FenceImpl*>(fence_.get());
          auto error = event.valid() ? event.error() : nullptr;
          std::unique_lock<std::mutex> lk(f.mtx);
          if (error && !f.error) {
            f.error = std::move(error);
          }
          f.value = count;
          f.cv.notify_all();
        });
  } else {
    throw std::runtime_error("[Fence::update] Invalid stream.");
  }
}

} // namespace mlx::core
