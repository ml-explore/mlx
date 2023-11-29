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

TEST_CASE("test eval with tracer") {
  auto x = array(1);
  x.set_tracer(true);

  // Ok, x is not a node
  eval(x);

  x = ones({2, 3});
  x.set_tracer(true);
  CHECK_THROWS(eval(x));

  // Ok retain_graph=true
  eval({x}, true);

  // Make sure all arguments are checked
  auto y = ones({2, 3});
  CHECK_THROWS(eval(x, y));
}

TEST_CASE("test eval graph retention") {
  auto x = array(1);
  auto y = array(2);
  auto z = x + y;
  eval({z}, true);
  CHECK(z.has_primitive());
  CHECK(z.is_evaled());
  CHECK_EQ(z.item<int>(true), 3);
  CHECK(z.has_primitive());
  CHECK(z.is_evaled());

  CHECK_EQ(z.item<int>(), 3);
  CHECK(!z.has_primitive());
  CHECK(z.is_evaled());

  z = x + y;
  auto a = z + x;
  auto b = a + y;
  eval({b}, true);
  CHECK(z.has_primitive());
  CHECK(z.is_evaled());
  CHECK(a.has_primitive());
  CHECK(a.is_evaled());

  eval({b}, false);
  CHECK(!z.has_primitive());
  CHECK(z.is_evaled());
  CHECK(!a.has_primitive());
  CHECK(a.is_evaled());
}
