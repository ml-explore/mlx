// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test simplify scalars") {
  auto a = array({-1.0f, 2.0f});
  auto b = maximum(a, array(0.0f));
  auto c = maximum(-a, array(0.0f));
  auto d = b + c;
  simplify({d});
  CHECK(b.inputs()[1].id() == c.inputs()[1].id());
}

TEST_CASE("test simplify") {
  auto a = array({1.0f, 2.0f});
  auto b = exp(a) + exp(a);
  simplify(b);
  eval(b);
  CHECK(b.inputs()[0].id() == b.inputs()[1].id());
}

TEST_CASE("test no simplify") {
  auto a = array({1.0f, 2.0f});
  auto b = cos(a) + sin(a);
  simplify(b);
  eval(b);
  CHECK(b.inputs()[0].id() != b.inputs()[1].id());
}
