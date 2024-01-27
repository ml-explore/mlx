// Copyright © 2023-2024 Apple Inc.

#include <iostream>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test simple custom vjp") {
  auto one = array(1.0);
  auto x = array(2.0);
  auto y = array(3.0);

  auto fn = [](const std::vector<array>& inputs) {
    return std::vector<array>{inputs[0] * inputs[1], inputs[0] + inputs[1]};
  };
  auto wrapped = [&fn, &one](const std::vector<array>& inputs) {
    return custom_vjp(
        fn,
        [&](const std::vector<array>&,
            const std::vector<array>&,
            const std::vector<array>&) {
          return std::vector<array>{one, one};
        },
        inputs);
  };

  auto [z, g] = vjp(fn, {x, y}, {one, one});
  CHECK_EQ(z[0].item<float>(), 6.0f);
  CHECK_EQ(z[1].item<float>(), 5.0f);
  CHECK_EQ(g[0].item<float>(), 4.0f);
  CHECK_EQ(g[1].item<float>(), 3.0f);

  std::tie(z, g) = vjp(wrapped, {x, y}, {one, one});
  CHECK_EQ(z[0].item<float>(), 6.0f);
  CHECK_EQ(z[1].item<float>(), 5.0f);
  CHECK_EQ(g[0].item<float>(), 1.0f);
  CHECK_EQ(g[1].item<float>(), 1.0f);
}

TEST_CASE("test checkpointing") {
  auto one = array(1.0);
  auto x = array(2.0);
  auto y = array(3.0);

  int cnt = 0;
  auto fn = [&cnt](const std::vector<array>& inputs) {
    cnt++;
    auto x = inputs[0] * inputs[1];
    auto y = inputs[0] + inputs[1];
    return std::vector<array>{square(x + y)};
  };
  auto wrapped = [&fn, &one](const std::vector<array>& inputs) {
    return checkpoint(fn, inputs);
  };

  auto [z, g] = vjp(wrapped, {x, y}, {one});
  CHECK_EQ(z[0].item<float>(), 121.0f);
  CHECK_EQ(g[0].item<float>(), 88.0f);
  CHECK_EQ(g[1].item<float>(), 66.0f);
  CHECK_EQ(cnt, 2);
}
