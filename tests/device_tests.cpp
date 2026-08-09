// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <set>

#include "mlx/mlx.h"

using namespace mlx::core;

template <typename T, size_t N>
void check_strict_weak_ordering(const std::array<T, N>& ordered_values) {
  for (const auto& value : ordered_values) {
    CHECK_FALSE(value < value);
  }

  for (const auto& lhs : ordered_values) {
    for (const auto& rhs : ordered_values) {
      if (lhs < rhs) {
        CHECK_FALSE(rhs < lhs);
      }
      for (const auto& other : ordered_values) {
        if (lhs < rhs && rhs < other) {
          CHECK(lhs < other);
        }
      }
    }
  }

  std::set<T> values(ordered_values.begin(), ordered_values.end());
  REQUIRE_EQ(values.size(), ordered_values.size());
  CHECK(std::equal(values.begin(), values.end(), ordered_values.begin()));
}

TEST_CASE("test device and stream ordering") {
  const std::array devices{
      Device(Device::cpu, 0),
      Device(Device::cpu, 3),
      Device(Device::gpu, 0),
      Device(Device::gpu, 1),
      Device(Device::gpu, 2)};
  check_strict_weak_ordering(devices);

  const std::array streams{
      Stream(0, Device::cpu),
      Stream(3, Device::cpu),
      Stream(0, Device::gpu),
      Stream(1, Device::gpu),
      Stream(2, Device::gpu)};
  check_strict_weak_ordering(streams);
}

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
