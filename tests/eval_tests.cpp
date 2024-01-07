// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test eval") {
  {
    array x(1.0);
    array y(1);
    array z(true);
    eval({x, y, z});
    CHECK_EQ(x.item<float>(), 1.0);
  }

  {
    array x(1.0);
    array y = ones({2, 2});
    array z(true);
    eval({x, y, z});
    CHECK(array_equal(y, array({1.0, 1.0, 1.0, 1.0}, {2, 2})).item<bool>());
  }
}

TEST_CASE("test eval multiple") {
  auto x = ones({10, 10});
  auto y = ones({10, 10});
  eval({x, y});
  CHECK(array_equal(x, y).item<bool>());

  auto a = x + y;
  auto b = x - y;
  eval({a, b});
  CHECK(array_equal(a, full({10, 10}, 2.0f)).item<bool>());
  CHECK(array_equal(b, full({10, 10}, 0.0f)).item<bool>());

  x = ones({10, 10});
  y = ones({10, 10});
  eval(x, y);
  CHECK(array_equal(x, y).item<bool>());

  a = x + y;
  b = x - y;
  eval(a, b);
  CHECK(array_equal(a, full({10, 10}, 2.0f)).item<bool>());
  CHECK(array_equal(b, full({10, 10}, 0.0f)).item<bool>());
}

TEST_CASE("test eval with tracer when not tracing") {
  // Since we are not tracing it doesn't matter that the array flags are
  // tracers they will always be detached.
  auto x = array(1);
  x.set_tracer(true);
  CHECK(!x.is_tracer());
  eval(x);
  CHECK(!x.has_primitive());
  CHECK(x.is_evaled());

  x = ones({2, 3});
  x.set_tracer(true);
  eval(x);
  CHECK(!x.has_primitive());
  CHECK(x.is_evaled());
}

TEST_CASE("test eval graph retention when not tracing") {
  // Since we are not tracing it doesn't matter that the array flags are
  // tracers they will always be detached.
  auto x = array(1);
  x.set_tracer(true);
  auto y = array(2);
  auto z = x + y;
  eval(z);
  CHECK(!z.has_primitive());
  CHECK(z.is_evaled());
  CHECK_EQ(z.item<int>(), 3);

  z.set_tracer(false);
  CHECK_EQ(z.item<int>(), 3);
  CHECK(!z.has_primitive());
  CHECK(z.is_evaled());

  z = x + y;
  auto a = z + x;
  auto b = a + y;
  eval(b);
  CHECK(!z.has_primitive());
  CHECK(z.is_evaled());
  CHECK(!a.has_primitive());
  CHECK(a.is_evaled());
}
