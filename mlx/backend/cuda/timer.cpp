// Copyright © 2026 Apple Inc.

#include "mlx/timer.h"

#include <stdexcept>

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/event.h"

namespace mlx::core {

namespace {

struct TimerImpl {
  explicit TimerImpl(cu::Device& device)
      : start(device, cudaEventBlockingSync),
        end(device, cudaEventBlockingSync) {}

  cu::CudaEvent start;
  cu::CudaEvent end;
  bool started{false};
  bool stopped{false};
};

} // namespace

Timer::Timer(Stream stream) : stream_(stream) {
  if (stream.device != Device::gpu) {
    throw std::invalid_argument("[Timer] GPU timing is not supported on CPU.");
  }
  impl_ = std::shared_ptr<void>(
      new TimerImpl{cu::device(stream.device)},
      [](void* ptr) { delete static_cast<TimerImpl*>(ptr); });
}

void Timer::start(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  auto& impl = *static_cast<TimerImpl*>(impl_.get());
  if (impl.started) {
    throw std::runtime_error("[Timer] The timer has already started.");
  }

  auto& encoder = cu::get_command_encoder(stream_);
  for (const auto& input : inputs) {
    encoder.set_input_array(input);
  }
  for (const auto& output : outputs) {
    encoder.set_output_array(output);
  }
  encoder.add_event_record_node(impl.start.handle());
  impl.started = true;
}

void Timer::stop(
    const std::vector<array>& inputs,
    const std::vector<array>& outputs) {
  auto& impl = *static_cast<TimerImpl*>(impl_.get());
  if (!impl.started) {
    throw std::runtime_error("[Timer] The timer has not started.");
  }
  if (impl.stopped) {
    throw std::runtime_error("[Timer] The timer has already stopped.");
  }

  auto& encoder = cu::get_command_encoder(stream_);
  for (const auto& input : inputs) {
    encoder.set_input_array(input);
  }
  for (const auto& output : outputs) {
    encoder.set_output_array(output);
  }
  encoder.add_event_record_node(impl.end.handle());
  impl.stopped = true;
}

double Timer::elapsed_time() {
  auto& impl = *static_cast<TimerImpl*>(impl_.get());
  if (!impl.stopped) {
    throw std::runtime_error("[Timer] The timer has not completed.");
  }
  return impl.start.elapsed_time(impl.end);
}

} // namespace mlx::core
