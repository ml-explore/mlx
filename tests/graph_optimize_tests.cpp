// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test simplify scalars") {
  {
    auto a = array(-1.0f);
    auto b = array(-1.0f);
    auto c = abs(a);
    auto d = abs(b);
    simplify({c, d});
    CHECK(c.inputs()[0].id() == d.inputs()[0].id());
  }

  {
    auto a = array({-1.0f, 2.0f});
    auto b = maximum(a, array(0.0f));
    auto c = maximum(-a, array(0.0f));
    auto d = b + c;
    simplify({d});
    CHECK(b.inputs()[1].id() == c.inputs()[1].id());
  }
}

TEST_CASE("test simplify") {
  auto a = array({1.0f, 2.0f});
  auto b = exp(a) + exp(a);
  simplify(b);
  CHECK(b.inputs()[0].id() == b.inputs()[1].id());
}

TEST_CASE("test no simplify") {
  auto a = array({1.0f, 2.0f});
  auto b = cos(a) + sin(a);
  simplify(b);
  CHECK(b.inputs()[0].id() != b.inputs()[1].id());
}

TEST_CASE("test simplify multi output") {
  {
    auto a = array(1.0);
    auto b = array(2.0);
    auto c = divmod(a, b);
    auto d = divmod(a, b);
    auto e = c[0] + d[0];
    auto f = c[1] + d[1];

    simplify({e, f});
    CHECK_EQ(e.inputs()[0].id(), e.inputs()[1].id());
    CHECK_EQ(f.inputs()[0].id(), f.inputs()[1].id());
  }

  {
    auto a = array(1.0);
    auto b = array(1.0);
    auto c = divmod(a, b);
    simplify(c);
    CHECK_EQ(c[0].inputs()[0].id(), c[0].inputs()[1].id());
    CHECK_EQ(c[0].inputs()[0].id(), c[1].inputs()[0].id());
    CHECK_EQ(c[1].inputs()[0].id(), c[1].inputs()[1].id());
  }

  // Make sure the output order of multi-output primitives
  // is respected in simplification
  {
    auto a = array(1.0);
    auto b = array(2.0);
    auto c = divmod(a, b);
    auto d = divmod(a, b);
    auto e = stack({c[0], c[1], d[0], d[1]});
    simplify(e);
    CHECK(array_equal(e, array({0.0f, 1.0f, 0.0f, 1.0f})).item<bool>());
    CHECK_EQ(e.inputs()[0].id(), e.inputs()[2].id());
    CHECK_EQ(e.inputs()[1].id(), e.inputs()[3].id());
  }
}
