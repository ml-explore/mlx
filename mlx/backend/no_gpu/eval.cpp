// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/gpu/available.h"
#include "mlx/backend/gpu/eval.h"

namespace mlx::core::gpu {

bool is_available() {
  return false;
}

void new_stream(Stream) {}

void eval(array&) {
  throw std::runtime_error("[gpu::eval] GPU backend is not available");
}

void finalize(Stream) {
  throw std::runtime_error("[gpu::finalize] GPU backend is not available");
}

void synchronize(Stream) {
  throw std::runtime_error("[gpu::synchronize]  GPU backend is not available");
}

} // namespace mlx::core::gpu
