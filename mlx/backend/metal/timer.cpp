// Copyright © 2026 Apple Inc.

#include "mlx/timer.h"

#include <stdexcept>

#include "mlx/backend/metal/device.h"

namespace mlx::core {

namespace {

struct TimerImpl {
  ~TimerImpl() {
    auto pool = metal::new_scoped_memory_pool();
    start.reset();
    end.reset();
  }

  NS::SharedPtr<MTL::CommandBuffer> start;
  NS::SharedPtr<MTL::CommandBuffer> end;
};

void check_command_buffer(MTL::CommandBuffer* command_buffer) {
  if (command_buffer->status() == MTL::CommandBufferStatusError) {
    auto* error = command_buffer->error();
    auto description = error ? error->localizedDescription()->utf8String()
                             : "unknown Metal error";
    throw std::runtime_error(
        std::string("[Timer] Command buffer execution failed: ") + description +
        ".");
  }
}

} // namespace

Timer::Timer(Stream stream) : stream_(stream) {
  if (stream.device != Device::gpu) {
    throw std::invalid_argument("[Timer] GPU timing is not supported on CPU.");
  }
  impl_ = std::shared_ptr<void>(
      new TimerImpl{}, [](void* ptr) { delete static_cast<TimerImpl*>(ptr); });
}

void Timer::start(const std::vector<array>&, const std::vector<array>&) {
  auto& impl = *static_cast<TimerImpl*>(impl_.get());
  if (impl.start) {
    throw std::runtime_error("[Timer] The timer has already started.");
  }

  auto& encoder = metal::get_command_encoder(stream_);
  encoder.end_encoding();
  encoder.commit();
  impl.start = NS::RetainPtr(encoder.get_command_buffer());
}

void Timer::stop(const std::vector<array>&, const std::vector<array>&) {
  auto& impl = *static_cast<TimerImpl*>(impl_.get());
  if (!impl.start) {
    throw std::runtime_error("[Timer] The timer has not started.");
  }
  if (impl.end) {
    throw std::runtime_error("[Timer] The timer has already stopped.");
  }

  auto& encoder = metal::get_command_encoder(stream_);
  impl.end = NS::RetainPtr(encoder.get_command_buffer());
  encoder.end_encoding();
  encoder.commit();
}

double Timer::elapsed_time() {
  auto& impl = *static_cast<TimerImpl*>(impl_.get());
  if (!impl.end) {
    throw std::runtime_error("[Timer] The timer has not completed.");
  }

  auto pool = metal::new_scoped_memory_pool();
  impl.end->waitUntilCompleted();
  check_command_buffer(impl.start.get());
  check_command_buffer(impl.end.get());

  auto start = impl.start->GPUStartTime();
  auto end = impl.end->GPUEndTime();
  if (start == 0.0 || end == 0.0 || end < start) {
    throw std::runtime_error("[Timer] Metal GPU timing is unavailable.");
  }
  return (end - start) * 1e3;
}

} // namespace mlx::core
