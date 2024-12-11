// Copyright © 2024 Apple Inc.

#include <cassert>
#include <iostream>

#include "mlx/mlx.h"

namespace mx = mlx::core;

int main() {
  // To use Metal debugging and profiling:
  // 1. Build with the MLX_METAL_DEBUG CMake option (i.e. -DMLX_METAL_DEBUG=ON).
  // 2. Run with MTL_CAPTURE_ENABLED=1.
  mx::metal::start_capture("mlx_trace.gputrace");

  // Start at index two because the default GPU and CPU streams have indices
  // zero and one, respectively. This naming matches the label assigned to each
  // stream's command queue.
  auto s2 = new_stream(mx::Device::gpu);
  auto s3 = new_stream(mx::Device::gpu);

  auto a = mx::arange(1.f, 10.f, 1.f, mx::float32, s2);
  auto b = mx::arange(1.f, 10.f, 1.f, mx::float32, s3);
  auto x = mx::add(a, a, s2);
  auto y = mx::add(b, b, s3);

  // The multiply will happen on the default stream.
  std::cout << mx::multiply(x, y) << std::endl;

  mx::metal::stop_capture();
}
