// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include <cstdlib>

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test device placement") {
  auto device = default_device();
  Device d = gpu::is_available() ? Device::gpu : Device::cpu;
  if (std::getenv("DEVICE") == nullptr) {
    CHECK_EQ(device, d);
  }

  array x(1.0f);
  array y(1.0f);
  auto z = add(x, y, default_device());
  if (gpu::is_available()) {
    z = add(x, y, Device::gpu);
    z = add(x, y, Device(Device::gpu, 0));
  } else {
    CHECK_THROWS_AS(set_default_device(Device::gpu), std::invalid_argument);
    CHECK_THROWS_AS(add(x, y, Device::gpu), std::invalid_argument);
  }

  // Set the default device to the CPU
  set_default_device(Device::cpu);
  CHECK_EQ(default_device(), Device::cpu);

  // Revert
  set_default_device(device);
}
