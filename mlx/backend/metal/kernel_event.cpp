// Copyright © 2024 Apple Inc.
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal_impl.h"
#include "mlx/event.h"

namespace mlx::core {

uint32_t* get_ptr(const Event& e) {
  return static_cast<uint32_t*>(
      static_cast<MTL::Buffer*>(e.raw_event().get())->contents());
}

void dispatch_signal_or_wait(Event e, bool signal) {
  auto& d = metal::device(e.stream().device);
  auto idx = e.stream().index;
  auto& compute_encoder = d.get_command_encoder(idx);
  auto kernel = d.get_kernel(signal ? "event_signal" : "event_wait");
  MTL::Size kernel_dims = MTL::Size(1, 1, 1);
  compute_encoder.set_compute_pipeline_state(kernel);
  auto buf = static_cast<MTL::Buffer*>(e.raw_event().get());
  if (signal) {
    // no kernels should start before this one is done
    d.barrier(idx);
  }
  compute_encoder.set_buffer(buf, 0);
  auto val = static_cast<uint32_t>(e.value());
  compute_encoder.set_bytes(val, 1);
  compute_encoder.dispatch_threads(kernel_dims, kernel_dims);
  if (!signal) {
    // no kernels should start before this one is done
    // insert encoder barrier for inside command encoder
    // update event fence for between command encoders
    compute_encoder.barrier();
    compute_encoder.update_fence(d.get_event_fence(idx));
  }
  auto command_buffer = d.get_command_buffer(idx);
  command_buffer->addCompletedHandler(
      [e = std::move(e)](MTL::CommandBuffer* cbuf) {});
}

void encode_wait(Event e) {
  dispatch_signal_or_wait(std::move(e), false);
}

void encode_signal(Event e) {
  dispatch_signal_or_wait(std::move(e), true);
}

Event::Event(const Stream& stream) : stream_(stream) {
  auto dtor = [](void* ptr) {
    allocator::free(static_cast<MTL::Buffer*>(ptr));
  };

  auto buf = allocator::malloc_or_wait(sizeof(uint32_t)).ptr();
  event_ = std::shared_ptr<void>(buf, dtor);
  *get_ptr(*this) = 0;
}

void Event::wait() {
  uint32_t* volatile ptr = get_ptr(*this);
  while (*ptr < value()) {
  }
}

void Event::signal() {
  uint32_t* volatile ptr = get_ptr(*this);
  *ptr = value();
}

bool Event::is_signaled() const {
  return *get_ptr(*this) >= value();
}

} // namespace mlx::core
