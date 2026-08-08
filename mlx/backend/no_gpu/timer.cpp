// Copyright © 2026 Apple Inc.

#include "mlx/timer.h"

#include <stdexcept>

namespace mlx::core {

Timer::Timer(Stream stream) : stream_(stream) {
  throw std::invalid_argument("[Timer] GPU timing requires a GPU backend.");
}

void Timer::start(const std::vector<array>&, const std::vector<array>&) {
  throw std::runtime_error("[Timer] GPU timing requires a GPU backend.");
}

void Timer::stop(const std::vector<array>&, const std::vector<array>&) {
  throw std::runtime_error("[Timer] GPU timing requires a GPU backend.");
}

double Timer::elapsed_time() {
  throw std::runtime_error("[Timer] GPU timing requires a GPU backend.");
}

} // namespace mlx::core
