// Copyright © 2024 Apple Inc.

#include <cassert>
#include <iostream>

#include "mlx/mlx.h"

using namespace mlx::core;

int main() {
  // Enable the MLX_METAL_DEBUG CMake option to enhance the capture with groups,
  // labels, etc.
  assert(metal::start_capture());

  // Start at index two because the default GPU and CPU streams have indices
  // zero and one, respectively. This naming matches the label assigned to each
  // stream's command queue.
  auto s2 = new_stream(Device::gpu);
  auto s3 = new_stream(Device::gpu);

  auto a = arange(1.f, 10.f, 1.f, float32, s2);
  auto b = arange(1.f, 10.f, 1.f, float32, s3);
  auto x = add(a, a, s2);
  auto y = add(b, b, s3);

  // The multiply will happen on the default stream.
  std::cout << multiply(x, y) << std::endl;

  metal::stop_capture();
}
