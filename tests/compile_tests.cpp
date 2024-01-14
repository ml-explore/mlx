// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

std::vector<array> simple_fun(const std::vector<array>& inputs) {
  return std::vector<array>{inputs[0] + inputs[1]};
};

TEST_CASE("test simple compile") {
  auto compfn = compile(simple_fun);
  auto out = compfn({array(1.0), array(2.0)})[0];
  CHECK_EQ(out.item<float>(), 3.0f);

  out = compfn({array(1.0), array(2.0)})[0];
  CHECK_EQ(out.item<float>(), 3.0f);
}
