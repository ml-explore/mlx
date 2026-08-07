// Copyright © 2026 Apple Inc.

#include "mlx/timer.h"

#include <algorithm>
#include <stdexcept>

#include "mlx/primitives.h"

namespace mlx::core {

namespace {

enum class TimerMarkerType { start, stop };

class TimerMarker : public Primitive {
 public:
  TimerMarker(Stream stream, Timer timer, TimerMarkerType type)
      : Primitive(stream), timer_(std::move(timer)), type_(type) {}

  void eval_cpu(const std::vector<array>&, std::vector<array>&) override {
    throw std::invalid_argument("[Timer] GPU timing is not supported on CPU.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    for (size_t i = 0; i < outputs.size(); ++i) {
      outputs[i].copy_shared_buffer(inputs[i]);
    }
    if (type_ == TimerMarkerType::start) {
      timer_.start(inputs, outputs);
    } else {
      timer_.stop(inputs, outputs);
    }
  }

  const char* name() const override {
    return type_ == TimerMarkerType::start ? "TimerStart" : "TimerStop";
  }

 private:
  Timer timer_;
  TimerMarkerType type_;
};

std::vector<array> timer_marker(
    const std::vector<array>& inputs,
    const Timer& timer,
    TimerMarkerType type) {
  if (inputs.empty()) {
    throw std::invalid_argument(
        "[Timer] Expected at least one array at each timing marker.");
  }
  if (std::any_of(inputs.begin(), inputs.end(), [](const array& input) {
        return input.is_tracer();
      })) {
    throw std::invalid_argument(
        "[Timer] Timers cannot be used inside graph transformations.");
  }

  std::vector<Shape> shapes;
  std::vector<Dtype> dtypes;
  shapes.reserve(inputs.size());
  dtypes.reserve(inputs.size());
  for (const auto& input : inputs) {
    shapes.push_back(input.shape());
    dtypes.push_back(input.dtype());
  }

  return array::make_arrays(
      std::move(shapes),
      dtypes,
      std::make_shared<TimerMarker>(timer.stream(), timer, type),
      inputs);
}

} // namespace

std::vector<array> timer_start(
    const std::vector<array>& inputs,
    const Timer& timer) {
  return timer_marker(inputs, timer, TimerMarkerType::start);
}

std::vector<array> timer_stop(
    const std::vector<array>& inputs,
    const Timer& timer) {
  return timer_marker(inputs, timer, TimerMarkerType::stop);
}

} // namespace mlx::core
