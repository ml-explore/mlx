// Copyright © 2026 Apple Inc.

#include <stdexcept>
#include <string>

#include "doctest/doctest.h"

#include "mlx/backend/metal/device.h"
#include "mlx/config.h"
#include "mlx/mlx.h"

using namespace mlx::core;

namespace {

constexpr auto source = R"(
[[kernel]] void fill_one(device float* out [[buffer(0)]]) {
  out[0] = 1.0f;
}
)";

} // namespace

TEST_CASE("test metal runtime compilation can be disabled") {
  auto& d = metal::device(Device::gpu);
  bool built = false;
  auto builder = [&built]() {
    built = true;
    return std::string(source);
  };

  auto cached = d.get_library("test_runtime_compile_cached", builder);
  REQUIRE(cached != nullptr);
  REQUIRE(built);

  {
    auto disabled = config::scoped_update({{"MLX_METAL_RUNTIME_COMPILE", 0}});
    built = false;
    CHECK_EQ(d.get_library("test_runtime_compile_cached", builder), cached);
    CHECK_THROWS_AS(
        d.get_library("test_runtime_compile_missing", builder),
        std::runtime_error);
    CHECK_FALSE(built);
  }

  CHECK(d.get_library("test_runtime_compile_missing", builder) != nullptr);
  CHECK(built);
}
