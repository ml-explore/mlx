// Copyright © 2023 Apple Inc.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <vector>
#include "doctest/doctest.h"

#include "mlx/graph_utils.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test stop gradient") {
  auto x = zeros({5, 5});
  auto y = stop_gradient(x);
  CHECK(array_equal(y, zeros({5, 5})).item<bool>());

  x = zeros({5, 5}, int32);
  y = stop_gradient(x);
  CHECK_EQ(y.dtype(), int32);
  CHECK(array_equal(y, zeros({5, 5}, int32)).item<bool>());

  {
    auto fun = [](array input) { return stop_gradient(add(input, ones({2}))); };
    auto vfun = vmap(fun);
    auto out = vfun(ones({3, 2}));
    CHECK(array_equal(out, full({3, 2}, 2.0)).item<bool>());
  }

  {
    auto fun = [](array input) { return add(stop_gradient(input), ones({2})); };
    auto vfun = vmap(fun);
    auto out = vfun(ones({3, 2}));
    CHECK(array_equal(out, full({3, 2}, 2.0)).item<bool>());
  }

  {
    auto x = array(1.);
    auto fun = [](array in) { return stop_gradient(add(in, in)); };
    auto out = vjp(fun, x, array(1.)).second;
    CHECK(array_equal(out, array(0.)).item<bool>());

    out = jvp(fun, x, array(1.)).second;
    CHECK(array_equal(out, array(0.)).item<bool>());
  }

  {
    auto x = array(1.);
    auto fun = [](array in) { return add(in, stop_gradient(in)); };
    auto out = vjp(fun, x, array(1.)).second;
    CHECK(array_equal(out, array(1.)).item<bool>());

    out = jvp(fun, x, array(1.)).second;
    CHECK(array_equal(out, array(1.)).item<bool>());
  }

  {
    auto x = array(1.);
    auto fun = [](array in) {
      for (int i = 0; i < 10; ++i) {
        in = add(in, in);
      }
      return stop_gradient(in);
    };
    {
      auto out = vjp(fun, x, array(1.)).second;
      std::ostringstream g_ss;
      print_graph(g_ss, out);
      auto g_str = g_ss.str();
      auto count = std::count(g_str.begin(), g_str.end(), '\n');
      CHECK(count < 5);
    }
    {
      auto out = jvp(fun, x, array(1.)).second;
      std::ostringstream g_ss;
      print_graph(g_ss, out);
      auto g_str = g_ss.str();
      auto count = std::count(g_str.begin(), g_str.end(), '\n');
      CHECK(count < 5);
    }
  }
}

TEST_CASE("test jvp") {
  {
    auto fun = [](const std::vector<array>& inputs) {
      return std::vector<array>{add(inputs[0], inputs[1])};
    };
    auto x = array(1.0f);
    auto y = array(1.0f);
    auto [out, dout] = jvp(fun, {x, y}, {array(1.0f), array(3.0f)});
    CHECK_EQ(out[0].item<float>(), 2.0f);
    CHECK_EQ(dout[0].item<float>(), 4.0f);
  }

  // Evaling in function while tracing performs graph retention
  {
    auto fun1 = [](const array& x) {
      auto y = 3 * x;
      eval(y);
      CHECK(y.is_evaled());
      CHECK(y.has_primitive());
      CHECK(y.is_tracer());
      return 2 * y;
    };
    CHECK_EQ(jvp(fun1, array(1.0f), array(1.0f)).second.item<float>(), 6.0f);
  }

  // Only one argument
  {
    auto x = array(1.0f);
    auto fun = [x](array in) { return add(x, in); };
    auto y = array(1.0f);
    auto out = jvp(fun, y, array(3.0f)).second;
    CHECK_EQ(out.item<float>(), 3.0f);
  }

  // Input also in capture clause
  {
    auto x = array(1.0f);
    auto fun = [x](array in) { return in + x; };
    auto out = jvp(fun, x, array(1.0f)).second;
    CHECK_EQ(out.item<float>(), 1.0f);
  }

  // Throws on incorrectly shaped inputs
  {
    auto fun = [](array in) { return add(in, in); };
    CHECK_THROWS_AS(jvp(fun, array(1), array({1, 1})), std::invalid_argument);
  }

  // Throws on wrong number of inputs
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{inputs[0], inputs[1]};
    };
    CHECK_THROWS_AS(
        jvp(fun, {array(1), array(1)}, {array(1)}), std::invalid_argument);
  }

  // No dependence between input and output
  {
    auto fun = [](array in) { return array({1.0, 1.0}); };
    auto out = jvp(fun, array(1.0f), array(1.0f)).second;
    CHECK(array_equal(out, zeros({2})).item<bool>());
  }
}

TEST_CASE("test vjp") {
  {
    auto x = array(1.0f);
    auto y = array(1.0f);
    auto fun = [y](array in) { return add(in, y); };
    auto [out, dout] = vjp(fun, x, array(1.0f));
    CHECK_EQ(out.item<float>(), 2.0f);
    CHECK_EQ(dout.item<float>(), 1.0f);
  }

  {
    auto x = array(1.0f);
    auto fun = [](array in) { return in + in + in; };
    auto out = vjp(fun, x, array(1.0f)).second;
    CHECK_EQ(out.item<float>(), 3.0f);
    out = vjp(fun, x, array(2.)).second;
    CHECK_EQ(out.item<float>(), 6.0f);
  }

  // Input also in capture clause
  {
    auto x = array(1.0f);
    auto fun = [x](array in) { return in + x; };
    auto out = vjp(fun, x, array(1.0f)).second;
    CHECK_EQ(out.item<float>(), 1.0f);
  }

  // Throws on incorrectly shaped outputs
  {
    auto fun = [](array in) { return add(in, in); };
    CHECK_THROWS_AS(vjp(fun, zeros({1}), zeros({2})), std::invalid_argument);
  }

  // Throws on wrong number of outputs
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{inputs[0], inputs[0]};
    };
    CHECK_THROWS_AS(
        vjp(fun, {zeros({1})}, {zeros({2})}), std::invalid_argument);
  }

  // No dependence between input and output
  {
    auto fun = [](array in) { return array(1.); };
    auto out = vjp(fun, zeros({2}), array(1.)).second;
    CHECK(array_equal(out, zeros({2})).item<bool>());
  }

  // Handles multiple outputs
  {
    auto x = array(1.);
    auto y = array(2.);
    auto z = array(3.);
    auto fun = [](const std::vector<array>& in) {
      return std::vector<array>{in[0] * in[1], in[1] * in[2]};
    };
    auto out = vjp(fun, {x, y, z}, {array(2.), array(3.)}).second;
    CHECK_EQ(out.size(), 3);
    CHECK_EQ(out[0].item<float>(), 2.0f * 2.0f);
    CHECK_EQ(out[1].item<float>(), 1.0f * 2.0f + 3.0f * 3.0f);
    CHECK_EQ(out[2].item<float>(), 3.0f * 2.0f);
  }
}

TEST_CASE("test grad") {
  {
    auto x = array(1.0);
    auto fun = [](array in) { return in + 1; };
    auto [y, dfdx] = value_and_grad(fun)(x);
    CHECK_EQ(y.item<float>(), 2.0f);
    CHECK_EQ(dfdx.item<float>(), 1.0f);
    auto [z, d2fdx2] = value_and_grad(grad(fun))(x);
    CHECK_EQ(z.item<float>(), 1.0f);
    CHECK_EQ(d2fdx2.item<float>(), 0.0f);
  }

  {
    auto x = array(1.);
    auto fun = [](array in) { return add(in, array(1.)); };
    auto dfdx = grad(fun);
    CHECK(array_equal(dfdx(x), array(1.)).item<bool>());
    auto d2fdx2 = grad(grad(fun));
    CHECK(array_equal(d2fdx2(x), array(0.)).item<bool>());
  }

  {
    auto x = array(1.);
    auto expfn = [](array input) { return exp(input); };
    auto dfdx = grad(expfn);
    CHECK_EQ(dfdx(x).item<float>(), doctest::Approx(std::exp(1.0f)));
    auto d2fdx2 = grad(grad(expfn));
    CHECK_EQ(d2fdx2(x).item<float>(), doctest::Approx(std::exp(1.0f)));
    auto d3fdx3 = grad(grad(grad(expfn)));
    CHECK_EQ(d3fdx3(x).item<float>(), doctest::Approx(std::exp(1.0f)));
  }

  {
    // No graph retention since the output is independent of y
    auto y = ones({3, 3});
    auto fn1 = [y](array x) {
      x = x + 2.0f;
      eval(y);
      CHECK(x.is_tracer());
      CHECK(!y.is_tracer());
      CHECK(y.is_evaled());
      CHECK(!y.has_primitive());
      return square(x);
    };
    auto dfdx = grad(fn1)(array(1.0f));
    CHECK_EQ(dfdx.item<float>(), 6.0f);

    // Graph automatically retained to compute the grad
    auto fn2 = [](array x) {
      x = x + 2.0f;
      eval(x);
      CHECK(x.is_tracer());
      CHECK(x.is_evaled());
      CHECK(x.has_primitive());
      return square(x);
    };
    dfdx = grad(fn2)(array(1.0f));
    CHECK_EQ(dfdx.item<float>(), 6.0f);
  }

  // Control flow in grad computation
  {
    auto fn = [](array x) {
      x = x + array(2.0f);
      if (x.item<float>() > 3) {
        return square(x);
      } else {
        return 4 * x;
      }
    };

    auto dfdx = grad(fn)(array(0.5f));
    CHECK_EQ(dfdx.item<float>(), 4.0f);

    dfdx = grad(fn)(array(1.5f));
    CHECK_EQ(dfdx.item<float>(), 7.0f);
  }

  // Grad with multiple inputs
  {
    auto fn = [](std::vector<array> inputs) { return inputs[0] * inputs[1]; };
    auto x = array(2.0f);
    auto y = array(3.0f);

    auto [value, grads] = value_and_grad(fn)({x, y});
    CHECK_EQ(value.item<float>(), 6.0f);
    CHECK_EQ(grads[0].item<float>(), 3.0f);

    auto dfdx = grad(fn)({x, y})[0];
    CHECK_EQ(dfdx.item<float>(), 3.0f);

    auto dfdy = grad(fn, 1)({x, y})[0];
    CHECK_EQ(dfdy.item<float>(), 2.0f);

    // Negative indexing
    dfdy = grad(fn, -1)({x, y})[0];
    CHECK_EQ(dfdy.item<float>(), 2.0f);

    grads = grad(fn, {0, 1})({x, y});
    CHECK_EQ(grads[0].item<float>(), 3.0f);
    CHECK_EQ(grads[1].item<float>(), 2.0f);

    CHECK_THROWS_AS(
        grad(fn, std::vector<int>{})({x, y}), std::invalid_argument);
    CHECK_THROWS_AS(grad(fn, {0, 1, 2})({x, y}), std::invalid_argument);
    CHECK_THROWS_AS(grad(fn, {0, 0})({x, y}), std::invalid_argument);
    CHECK_THROWS_AS(grad(fn, -3)({x, y}), std::invalid_argument);
  }
}

TEST_CASE("test creation grads") {
  // Test astype
  {
    auto fn = [](array a) { return astype(a, int32); };
    auto x = ones({4, 4}, float32);
    auto out = vjp(fn, x, full({4, 4}, 2, int32)).second;
    CHECK_EQ(out.dtype(), float32);
    CHECK(array_equal(out, full({4, 4}, 2.0f)).item<bool>());

    out = jvp(fn, x, full({4, 4}, 2, float32)).second;
    CHECK_EQ(out.dtype(), int32);
    CHECK(array_equal(out, full({4, 4}, 2, int32)).item<bool>());
  }

  // Test full
  {
    auto full_fn = [](array a) { return full({5, 5, 2}, a); };
    auto x = ones({2}, float32);
    auto out = vjp(full_fn, x, full({5, 5, 2}, 2.0f)).second;
    CHECK(array_equal(out, array({50.0f, 50.0f})).item<bool>());

    out = jvp(full_fn, x, array({3.0f, 3.0f})).second;
    CHECK(array_equal(out, full({5, 5, 2}, 3.0f)).item<bool>());
  }
}

TEST_CASE("test op vjps") {
  // Test abs
  {
    auto out = vjp([](array in) { return abs(in); }, array(-5.0f), array(1.0f));
    CHECK_EQ(out.second.item<float>(), -1.0f);
  }

  // Test sign
  {
    auto out =
        vjp([](array in) { return sign(in); }, array(-5.0f), array(10.0f));
    CHECK_EQ(out.second.item<float>(), 0.0f);
  }

  // Test negate
  {
    auto out = vjp([](array in) { return -in; }, array(1.0), array(2.0));
    CHECK(array_equal(out.second, array(-2.)).item<bool>());
  }

  // Test square
  {
    auto out =
        vjp([](array in) { return square(in); }, array(2.0f), array(3.0f));
    CHECK_EQ(out.second.item<float>(), 12.0f);
  }

  // Test sqrt
  {
    auto out = vjp(
        [](array in) { return mlx::core::sqrt(in); }, array(4.0f), array(8.0f));
    CHECK_EQ(out.second.item<float>(), 2.0f);
  }

  // Test rsqrt
  {
    auto out =
        vjp([](array in) { return rsqrt(in); }, array(4.0f), array(8.0f));
    CHECK_EQ(out.second.item<float>(), -0.5f);
  }

  // Test exp
  {
    auto out = vjp([](array in) { return exp(in); }, array(1.0f), array(2.0f));
    CHECK_EQ(out.second.item<float>(), doctest::Approx(2.0f * std::exp(1.0f)));
  }

  // Test sin
  {
    auto out =
        vjp([](array input) { return sin(input); }, array(1.0f), array(1.0f));
    CHECK(out.second.item<float>() == doctest::Approx(std::cos(1.0f)));
  }

  // Test cos
  {
    auto out =
        vjp([](array input) { return cos(input); }, array(1.0f), array(1.0f));
    CHECK(out.second.item<float>() == doctest::Approx(-std::sin(1.0f)));
  }

  // Test log
  {
    auto out = vjp([](array in) { return log(in); }, array(2.0f), array(1.0f));
    CHECK_EQ(out.second.item<float>(), 0.5f);

    out = vjp([](array in) { return log(in); }, array(2.0f), array(2.0f));
    CHECK_EQ(out.second.item<float>(), 1.0f);
  }

  // Test log1p
  {
    auto out =
        vjp([](array in) { return log1p(in); }, array(1.0f), array(1.0f));
    CHECK_EQ(out.second.item<float>(), 0.5f);

    out = vjp([](array in) { return log1p(in); }, array(1.0f), array(2.0f));
    CHECK_EQ(out.second.item<float>(), 1.0f);
  }

  constexpr auto inf = std::numeric_limits<float>::infinity();

  // Test erf
  {
    auto out = vjp([](array in) { return erf(in); }, array(inf), array(1.0f));
    CHECK_EQ(out.second.item<float>(), 0.0f);

    out = vjp([](array in) { return erf(in); }, array(-inf), array(2.0f));
    CHECK_EQ(out.second.item<float>(), 0.0f);

    out = vjp([](array in) { return erf(in); }, array(0.0f), array(1.0f));
    CHECK_EQ(out.second.item<float>(), static_cast<float>(M_2_SQRTPI));
  }

  // Test erfinv
  {
    auto out =
        vjp([](array in) { return erfinv(in); }, array(1.0f), array(1.0f));
    CHECK_EQ(out.second.item<float>(), inf);

    out = vjp([](array in) { return erfinv(in); }, array(-1.0f), array(2.0f));
    CHECK_EQ(out.second.item<float>(), inf);

    out = vjp([](array in) { return erfinv(in); }, array(0.0f), array(1.0f));
    CHECK_EQ(out.second.item<float>(), static_cast<float>(1.0 / M_2_SQRTPI));
  }

  // Test sigmoid
  {
    auto out =
        vjp([](array in) { return sigmoid(in); }, array(0.0f), array(1.0f));
    CHECK_EQ(out.second.item<float>(), 0.25f);

    out = vjp([](array in) { return sigmoid(in); }, array(0.0f), array(2.0f));
    CHECK_EQ(out.second.item<float>(), 0.5f);
  }

  // Test add
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{inputs[0] + inputs[1]};
    };
    auto out = vjp(fun, {array(1.0), array(2.0)}, {array(3.0)}).second;
    CHECK_EQ(out[0].item<float>(), 3.0);
    CHECK_EQ(out[1].item<float>(), 3.0);

    // Check with broadcasting
    out = vjp(fun, {ones({3, 1}), ones({1, 2})}, {full({3, 2}, 2.0)}).second;
    CHECK(array_equal(out[0], full({3, 1}, 4.0)).item<bool>());
    CHECK(array_equal(out[1], full({1, 2}, 6.0)).item<bool>());
  }

  // Test subtract
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{inputs[0] - inputs[1]};
    };
    auto out = vjp(fun, {array(1.0), array(2.0)}, {array(3.0)}).second;
    CHECK_EQ(out[0].item<float>(), 3.0);
    CHECK_EQ(out[1].item<float>(), -3.0);

    // Check with broadcasting
    out = vjp(fun, {ones({3, 1}), ones({1, 2})}, {ones({3, 2})}).second;
    CHECK(array_equal(out[0], full({3, 1}, 2.0)).item<bool>());
    CHECK(array_equal(out[1], full({1, 2}, -3.0)).item<bool>());
  }

  // Test multiply
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{inputs[0] * inputs[1]};
    };
    auto out = vjp(fun, {array(4.0f), array(2.0f)}, {array(3.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 6.0f);
    CHECK_EQ(out[1].item<float>(), 12.0f);

    // Check with broadcasting
    out = vjp(fun, {full({3, 1}, 2.0f), full({1, 2}, 4.0f)}, {ones({3, 2})})
              .second;
    CHECK(array_equal(out[0], full({3, 1}, 8.0f)).item<bool>());
    CHECK(array_equal(out[1], full({1, 2}, 6.0)).item<bool>());
  }

  // Test divide
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{inputs[0] / inputs[1]};
    };
    auto out = vjp(fun, {array(4.0f), array(2.0f)}, {array(1.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 0.5f);
    CHECK_EQ(out[1].item<float>(), -1.0f);

    // Check with broadcasting
    out = vjp(fun, {full({3, 1}, 4.0f), full({1, 2}, 2.0f)}, {ones({3, 2})})
              .second;
    CHECK(array_equal(out[0], full({3, 1}, 1.0f)).item<bool>());
    CHECK(array_equal(out[1], full({1, 2}, -3.0f)).item<bool>());
  }

  // Test maximum
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{maximum(inputs[0], inputs[1])};
    };
    auto out = vjp(fun, {array(5.0f), array(2.0f)}, {array(2.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 2.0f);
    CHECK_EQ(out[1].item<float>(), 0.0f);

    out = vjp(fun, {array(2.0f), array(2.0f)}, {array(1.0f)}).second;
    auto out_a = out[0].item<float>();
    auto out_b = out[1].item<float>();
    // When inputs are equal at most one gradient is nonzero
    CHECK(
        ((out_a == 1.0f && out_b == 0.0f) || (out_a == 0.0f && out_b == 1.0f)));
  }

  // Test minimum
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{minimum(inputs[0], inputs[1])};
    };
    auto out = vjp(fun, {array(4.0f), array(2.0f)}, {array(2.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 0.0f);
    CHECK_EQ(out[1].item<float>(), 2.0f);

    out = vjp(fun, {array(2.0f), array(2.0f)}, {array(1.0f)}).second;
    auto out_a = out[0].item<float>();
    auto out_b = out[1].item<float>();
    CHECK(
        ((out_a == 1.0f && out_b == 0.0f) || (out_a == 0.0f && out_b == 1.0f)));
  }

  // Test logaddexp
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{logaddexp(inputs[0], inputs[1])};
    };

    constexpr auto inf = std::numeric_limits<float>::infinity();

    auto out = vjp(fun, {array(2.0), array(2.0f)}, {array(1.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 0.5f);
    CHECK_EQ(out[1].item<float>(), 0.5f);
    out = vjp(fun, {array(2.0), array(2.0f)}, {array(2.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 1.0f);
    CHECK_EQ(out[1].item<float>(), 1.0f);

    out = vjp(fun, {array(inf), array(2.0f)}, {array(1.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 1.0f);
    CHECK_EQ(out[1].item<float>(), 0.0f);

    out = vjp(fun, {array(-inf), array(2.0f)}, {array(1.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 0.0f);
    CHECK_EQ(out[1].item<float>(), 1.0f);

    out = vjp(fun, {array(-10.0f), array(-inf)}, {array(1.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 1.0f);
    CHECK_EQ(out[1].item<float>(), 0.0f);

    out = vjp(fun, {array(-inf), array(-inf)}, {array(1.0f)}).second;
    CHECK(std::isnan(out[0].item<float>()));
    CHECK(std::isnan(out[1].item<float>()));
  }

  // Test power
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{inputs[0] ^ inputs[1]};
    };
    auto out = vjp(fun, {array(4.0f), array(3.0f)}, {array(1.0f)}).second;
    CHECK_EQ(out[0].item<float>(), 48.0f);
    CHECK_EQ(out[1].item<float>(), std::log(4.0f) * 64.0f);
  }

  // Test sum
  {
    std::vector<int> axes;
    auto fun = [&axes](array input) { return sum(input, axes); };
    axes = {};
    auto out = vjp(fun, array(2.0f), array(3.0f)).second;
    CHECK_EQ(out.item<float>(), 3.0f);

    axes = {0};
    out = vjp(fun, array({}), array(3.0f)).second;
    CHECK_EQ(out.size(), 0);
    CHECK_EQ(out.shape(), std::vector<int>{0});

    axes = {0};
    out = vjp(fun, ones({2, 2, 2}), array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}))
              .second;
    auto expected =
        array({1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f}, {2, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());

    axes = {1};
    out = vjp(fun, ones({2, 2, 2}), array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}))
              .second;
    expected =
        array({1.0f, 2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 3.0f, 4.0f}, {2, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());

    axes = {2};
    out = vjp(fun, ones({2, 2, 2}), array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}))
              .second;
    expected =
        array({1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f}, {2, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());
  }

  // Test prod
  {
    std::vector<int> axes;
    auto fun = [&axes](array input) { return prod(input, axes); };
    axes = {};
    auto out = vjp(fun, array(2.0f), array(3.0f)).second;
    CHECK_EQ(out.item<float>(), 3.0f);

    axes = {0};
    out = vjp(fun,
              array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}),
              array(
                  {1.0f, 2.0f, 3.0f},
                  {
                      3,
                  }))
              .second;
    auto expected = array({4.0f, 10.0f, 18.0f, 1.0f, 4.0f, 9.0f}, {2, 3});
    CHECK(array_equal(out, expected).item<bool>());

    axes = {0, 1};
    out = vjp(fun,
              array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}),
              array(1.0f))
              .second;
    expected = array({720.0f, 360.0f, 240.0f, 180.0f, 144.0f, 120.0f}, {2, 3});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test gather and take grads") {
  // Check linear takes
  auto linear_f = [](array indices) {
    auto fun_linear = [&indices](array input) { return take(input, indices); };

    return fun_linear;
  };

  auto src = ones({4, 4});
  auto ind = array({0, 1, 2, 3}, uint32);
  auto out = vjp(linear_f(ind), src, ones({4})).second;
  auto out_1 = take(out, array({0}, uint32), 0);
  auto out_2 = take(out, array({1, 2, 3}, uint32), 0);
  CHECK(array_equal(out_1, ones({1, 4})).item<bool>());
  CHECK(array_equal(out_2, zeros({3, 4})).item<bool>());
  auto tangent = reshape(arange(16), {4, 4});
  out = jvp(linear_f(ind), src, tangent).second;
  CHECK(array_equal(out, array({0, 1, 2, 3})).item<bool>());

  src = ones({4});
  ind = array({0, 0, 0, 0}, uint32);
  out = vjp(linear_f(ind), src, ones({4})).second;
  out_1 = take(out, array({0}, uint32));
  CHECK_EQ(out_1.item<float>(), 4.0f);

  tangent = arange(4);
  out = jvp(linear_f(ind), src, tangent).second;
  CHECK(array_equal(out, array({0, 0, 0, 0})).item<bool>());

  // Check axis takes
  src = ones({4, 4});
  ind = array({0, 1, 2, 3}, uint32);

  auto fun = [&ind](array input) { return take(input, ind, 0); };

  out = vjp(fun, src, ones({4, 4})).second;
  CHECK(array_equal(out, src).item<bool>());

  out = jvp(fun, src, ones({4, 4})).second;
  CHECK(array_equal(out, src).item<bool>());

  // Check index throw
  auto fun_throw = [](std::vector<array> inputs) {
    return std::vector<array>{take(inputs[0], inputs[1])};
  };

  CHECK_THROWS_AS(
      vjp(fun_throw, {src, ind}, {ones({4, 4})}), std::invalid_argument);

  CHECK_THROWS_AS(
      jvp(fun_throw, {src, ind}, {ones({4, 4}), ind}), std::invalid_argument);
}

TEST_CASE("test slice grads") {
  std::vector<int> start = {5, 0, 0};
  std::vector<int> stop = {7, 2, 4};
  std::vector<int> strides = {1, 1, 1};

  auto fn = [&start, &stop, &strides](array input) {
    return slice(input, start, stop, strides);
  };

  auto src = ones({8, 8, 8});
  auto out = vjp(fn, src, ones({2, 2, 4})).second;
  CHECK_EQ(sum(out).item<float>(), 16.);

  out = jvp(fn, src, full({8, 8, 8}, 2.0f)).second;
  CHECK(array_equal(out, full({2, 2, 4}, 2.0f)).item<bool>());

  src = ones({4, 4});
  start = {2, 0};
  stop = {4, 4};
  strides = {1, 1};
  out = vjp(fn, src, ones({2, 4})).second;
  auto out_1 = take(out, array({0, 1}, uint32), 0);
  auto out_2 = take(out, array({2, 3}, uint32), 0);

  CHECK(array_equal(out_1, zeros({2, 4})).item<bool>());
  CHECK(array_equal(out_2, ones({2, 4})).item<bool>());

  start = {0, 0};
  stop = {4, 4};
  strides = {2, 2};
  auto cotangent = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  out = vjp(fn, src, cotangent).second;
  auto expected = astype(
      array({1, 0, 2, 0, 0, 0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 0}, {4, 4}), float32);
  CHECK(array_equal(out, expected).item<bool>());

  out = jvp(fn, src, ones({4, 4})).second;
  CHECK(array_equal(out, ones({2, 2})).item<bool>());

  // Empty slices.
  start = {0, 0};
  stop = {0, 4};
  cotangent = reshape(array({}), {0, 2});
  out = vjp(fn, src, cotangent).second;
  CHECK(array_equal(out, zeros({4, 4})).item<bool>());

  out = jvp(fn, src, ones({4, 4})).second;
  CHECK_EQ(out.size(), 0);
}

TEST_CASE("test min and max vjp") {
  // Test min
  {
    std::vector<int> axes;
    array in({});
    array v({});
    array expected({});
    array out({});
    auto fun = [&axes](array input) { return min(input, axes); };

    axes = {};
    in = array({2.0f});
    out = vjp(fun, array(2.0f), array(3.0f)).second;
    CHECK_EQ(out.item<float>(), 3.0f);

    axes = {0};
    in = reshape(array({1.0f, 2.0f, 2.0f, -1.0f}), {2, 2});
    v = array({3.0f, 7.0f});
    out = vjp(fun, in, v).second;
    expected = array({3.0f, 0.0f, 0.0f, 7.0f});
    expected = reshape(expected, {2, 2});
    CHECK(array_equal(out, expected).item<bool>());

    axes = {0, 2};
    in = reshape(
        array({1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f, 1.0f, 0.0f}), {2, 2, 2});
    v = array({3.0f, 7.0f});
    out = vjp(fun, in, v).second;
    expected = array({0.0f, 0.0f, 3.5f, 0.0f, 1.5f, 1.5f, 0.0f, 3.5f});
    expected = reshape(expected, {2, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());
  }

  // Test max
  {
    std::vector<int> axes;
    array in({});
    array v({});
    array expected({});
    array out({});
    auto fun = [&axes](array input) { return max(input, axes); };

    axes = {};
    in = array({2.0f});
    out = vjp(fun, array(2.0f), array(3.0f)).second;
    CHECK_EQ(out.item<float>(), 3.0f);

    axes = {0};
    in = reshape(array({1.0f, 2.0f, 2.0f, -1.0f}), {2, 2});
    v = array({3.0f, 7.0f});
    out = vjp(fun, in, v).second;
    expected = array({0.0f, 7.0f, 3.0f, 0.0f});
    expected = reshape(expected, {2, 2});
    CHECK(array_equal(out, expected).item<bool>());

    axes = {0, 2};
    in = reshape(
        array({1.0f, 0.0f, 0.0f, 1.0f, -1.0f, -1.0f, 1.0f, 0.0f}), {2, 2, 2});
    v = array({3.0f, 7.0f});
    out = vjp(fun, in, v).second;
    expected = array({3.0f, 0.0f, 0.0f, 3.5f, 0.0f, 0.0f, 3.5f, 0.0f});
    expected = reshape(expected, {2, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test reshape and transpose grads") {
  {
    auto fn = [](array a) { return reshape(a, {3, 4}); };

    auto out = vjp(fn, ones({12}), full({3, 4}, 2.0f)).second;
    CHECK(array_equal(out, full({12}, 2.0f)).item<bool>());

    out = jvp(fn, ones({12}), full({12}, 2.0f)).second;
    CHECK(array_equal(out, full({3, 4}, 2.0f)).item<bool>());
  }

  {
    auto fn = [](array a) { return transpose(a, {1, 2, 0}); };

    auto cotan = reshape(arange(24), {3, 4, 2});
    auto out = vjp(fn, ones({2, 3, 4}), cotan).second;
    CHECK(array_equal(out, transpose(cotan, {2, 0, 1})).item<bool>());

    auto tangent = reshape(arange(24), {2, 3, 4});
    out = jvp(fn, ones({2, 3, 4}), tangent).second;
    CHECK(array_equal(out, transpose(tangent, {1, 2, 0})).item<bool>());
  }
}

TEST_CASE("test copy grads") {
  auto fn = [](array a) { return copy(a); };

  auto cotan = arange(4, float32);
  auto out = vjp(fn, ones({4}), cotan).second;
  CHECK(array_equal(out, arange(4, float32)).item<bool>());

  auto tangent = arange(4, float32);
  out = jvp(fn, ones({4}), tangent).second;
  CHECK(array_equal(out, tangent).item<bool>());
}

TEST_CASE("test matmul vjp") {
  auto fun = [](std::vector<array> inputs) {
    return std::vector<array>{matmul(inputs[0], inputs[1])};
  };

  auto a = array({1.0f, 2.0f}, {1, 2});
  auto b = array({3.0f, 4.0f}, {2, 1});
  auto out = vjp(fun, {a, b}, {array({2.0f}, {1, 1})}).second;

  CHECK(array_equal(out[0], array({6.0f, 8.0f}, {1, 2})).item<bool>());
  CHECK(array_equal(out[1], array({2.0f, 4.0f}, {2, 1})).item<bool>());

  a = array({1.0f, 2.0f}, {2, 1});
  b = array({3.0f, 4.0f}, {1, 2});
  out = vjp(fun, {a, b}, {array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2})}).second;
  CHECK(array_equal(out[0], array({11.0f, 25.0f}, {2, 1})).item<bool>());
  CHECK(array_equal(out[1], array({7.0f, 10.0f}, {1, 2})).item<bool>());

  a = array({1.0f, 2.0f, 1.0f, 2.0f}, {2, 2, 1});
  b = array({1.0f, 1.0f, 2.0f, 2.0f}, {2, 1, 2});
  auto vjps = vjp(fun, {a, b}, {ones({2, 2, 2})}).second;
  auto vjpx = array({2.0f, 2.0f, 4.0f, 4.0f}, {2, 2, 1});
  auto vjpy = array({3.0f, 3.0f, 3.0f, 3.0f}, {2, 1, 2});
  CHECK(array_equal(vjps[0], vjpx).item<bool>());
  CHECK(array_equal(vjps[1], vjpy).item<bool>());
}

TEST_CASE("test concatenate grads") {
  auto arrs = split(arange(5, float32), 5);
  eval(arrs);

  auto fn = [&arrs](const std::vector<array>& inputs) {
    arrs[2] = inputs[0];
    arrs[4] = inputs[1];
    return std::vector<array>{concatenate(arrs, 0)};
  };
  auto out = vjp(fn, {arrs[2], arrs[4]}, {arange(5, float32)}).second;

  CHECK_EQ(out.size(), 2);
  CHECK_EQ(out[0].item<float>(), 2.0f);
  CHECK_EQ(out[1].item<float>(), 4.0f);

  out = jvp(fn, {arrs[2], arrs[4]}, {array({2.0f}, {1}), array({3.0f}, {1})})
            .second;
  CHECK_EQ(out.size(), 1);
  CHECK(
      array_equal(out[0], array({0.0f, 0.0f, 2.0f, 0.0f, 3.0f})).item<bool>());
}

TEST_CASE("test split grads") {
  array x = arange(6, float32);
  eval(x);

  {
    auto fn = [](const array& x) {
      auto parts = split(x, 3);
      return parts[0] * parts[1] + parts[2];
    };
    auto out = vjp(fn, {x}, {ones({2})}).second;

    CHECK_EQ(out.size(), 6);
    CHECK(array_equal(out, array({2.0f, 3.0f, 0.0f, 1.0f, 1.0f, 1.0f}))
              .item<bool>());
  }

  {
    auto fn = [](const array& x) {
      auto parts = split(x, 3);
      return parts[0] * parts[2];
    };
    auto out = vjp(fn, {x}, {ones({2})}).second;

    CHECK_EQ(out.size(), 6);
    CHECK(array_equal(out, array({4.0f, 5.0f, 0.0f, 0.0f, 0.0f, 1.0f}))
              .item<bool>());
  }
}

TEST_CASE("test comparison grads") {
  auto x = ones({3, 1});
  auto y = zeros({1, 3});

  auto check_vjp_jvp = [&x, &y](auto fn) {
    auto fn_wrap = [&fn](std::vector<array> inputs) {
      return std::vector<array>{fn(inputs[0], inputs[1], default_device())};
    };
    auto out_shape = broadcast_shapes(x.shape(), y.shape());
    std::vector<array> vjps = vjp(fn_wrap, {x, y}, {ones(out_shape)}).second;
    bool correct = array_equal(vjps[0], zeros(x.shape())).item<bool>();
    correct &= array_equal(vjps[1], zeros(y.shape())).item<bool>();

    std::vector<array> jvps =
        jvp(fn_wrap, {x, y}, {ones(x.shape()), ones(y.shape())}).second;
    correct &= array_equal(jvps[0], zeros(out_shape)).item<bool>();
    return correct;
  };

  CHECK(check_vjp_jvp(equal));
  CHECK(check_vjp_jvp(greater));
  CHECK(check_vjp_jvp(less));
  CHECK(check_vjp_jvp(greater_equal));
  CHECK(check_vjp_jvp(less_equal));
}

TEST_CASE("test as_strided grads") {
  auto x = ones({11});
  std::vector<int> shape = {5, 5};
  std::vector<size_t> strides = {1, 1};
  size_t offset = 0;

  auto fun = [&shape, &strides, &offset](array x) {
    return as_strided(x, shape, strides, offset);
  };

  auto out = vjp(fun, x, ones(shape)).second;
  auto expected = array({1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0});
  CHECK(array_equal(out, expected).item<bool>());

  offset = 1;
  out = vjp(fun, x, ones(shape)).second;
  expected = array({0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0});
  CHECK(array_equal(out, expected).item<bool>());

  offset = 3;
  shape = {3, 3};
  strides = {0, 1};
  out = vjp(fun, x, ones(shape)).second;
  expected = array({0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0});
  CHECK(array_equal(out, expected).item<bool>());

  offset = 3;
  shape = {3, 3};
  strides = {0, 1};
  out = vjp(fun, x, reshape(astype(arange(9), x.dtype()), {3, 3})).second;
  expected = array({0, 0, 0, 9, 12, 15, 0, 0, 0, 0, 0});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test jvp from vjp") {
  // Unary element-wise ops
  {
    auto x = random::uniform({5, 10});
    eval(x);

    auto compute_derivs = [&x](auto fn) {
      auto fn_wrap = [&fn](array input) { return fn(input, default_device()); };

      // Compute vjp
      array vjp_out = vjp(fn_wrap, x, ones(x.shape())).second;

      // Compute jvp
      array jvp_out = jvp(fn_wrap, x, ones(x.shape())).second;

      return array_equal(vjp_out, jvp_out).item<bool>();
    };

    CHECK(compute_derivs(mlx::core::abs));
    CHECK(compute_derivs(mlx::core::cos));
    CHECK(compute_derivs(mlx::core::erf));
    CHECK(compute_derivs(mlx::core::erfinv));
    CHECK(compute_derivs(mlx::core::exp));
    CHECK(compute_derivs(mlx::core::log));
    CHECK(compute_derivs(mlx::core::log1p));
    CHECK(compute_derivs(mlx::core::negative));
    CHECK(compute_derivs(mlx::core::sigmoid));
    CHECK(compute_derivs(mlx::core::sign));
    CHECK(compute_derivs(mlx::core::sin));
    CHECK(compute_derivs(mlx::core::square));
    CHECK(compute_derivs(mlx::core::sqrt));
    CHECK(compute_derivs(mlx::core::rsqrt));
  }

  // Binary element-wise ops
  {
    auto x = random::uniform({5, 10});
    auto y = random::uniform({5, 10});
    eval(x, y);

    auto compute_derivs = [&x, &y](auto fn) {
      auto fn_wrap = [&fn](std::vector<array> inputs) {
        return std::vector<array>{fn(inputs[0], inputs[1], default_device())};
      };

      // Compute vjp and add results
      auto vjps = vjp(fn_wrap, {x, y}, {ones(x.shape())}).second;
      array vjp_out = add(vjps[0], vjps[1]);

      // Compute jvp
      array jvp_out =
          jvp(fn_wrap, {x, y}, {ones(x.shape()), ones(y.shape())}).second[0];
      return array_equal(vjp_out, jvp_out).item<bool>();
    };

    CHECK(compute_derivs(add));
    CHECK(compute_derivs(divide));
    CHECK(compute_derivs(logaddexp));
    CHECK(compute_derivs(maximum));
    CHECK(compute_derivs(minimum));
    CHECK(compute_derivs(multiply));
    CHECK(compute_derivs(subtract));
    CHECK(compute_derivs(power));
  }

  // Conditional selection element-wise op
  {
    auto condition = random::randint(0, 2, {5, 10});
    auto x = random::uniform({5, 10});
    auto y = random::uniform({5, 10});
    eval(condition, x, y);

    auto compute_derivs = [&condition, &x, &y](auto fn) {
      auto fn_wrap = [&fn](std::vector<array> inputs) {
        return std::vector<array>{
            fn(inputs[0], inputs[1], inputs[2], default_device())};
      };

      // Compute vjp and add results
      auto vjps = vjp(fn_wrap, {condition, x, y}, {ones(x.shape())}).second;
      auto vjp_out = add(add(vjps[0], vjps[1]), vjps[2]);

      // Compute jvp
      array jvp_out =
          jvp(fn_wrap,
              {condition, x, y},
              {ones(condition.shape()), ones(y.shape()), ones(x.shape())})
              .second[0];

      array result = array_equal(vjp_out, jvp_out);
      return result.item<bool>();
    };

    CHECK(compute_derivs(where));
  }
}

TEST_CASE("test complex gradients") {
  {
    auto add_fn = [](std::vector<array> inputs) {
      return std::vector<array>{add(inputs[0], inputs[1], default_device())};
    };

    // Compute jvp
    auto x = array(complex64_t{1.0, 1.0});
    auto y = array(complex64_t{1.0, 1.0});
    auto x_tan = array(complex64_t{1.0, 2.0});
    auto y_tan = array(complex64_t{2.0, 1.0});
    auto jvp_out = jvp(add_fn, {x, y}, {x_tan, y_tan}).second;
    CHECK_EQ(jvp_out[0].item<complex64_t>(), complex64_t{3.0, 3.0});

    // Compute vjp
    auto cotan = array(complex64_t{3.0, 3.0});
    auto vjp_out = vjp(add_fn, {x, y}, {cotan}).second;
    CHECK_EQ(vjp_out[0].item<complex64_t>(), complex64_t{3.0, 3.0});
    CHECK_EQ(vjp_out[1].item<complex64_t>(), complex64_t{3.0, 3.0});
  }

  {
    // Compute jvp
    auto x = array(complex64_t{2.0, 4.0});
    auto y = array(3.0f);

    auto x_tan = array(complex64_t{1.0, 2.0});
    auto y_tan = array(2.0f);

    auto out = jvp([x](array a) { return multiply(a, x); }, y, y_tan).second;
    CHECK_EQ(out.item<complex64_t>(), complex64_t{4.0, 8.0});

    out = jvp([y](array a) { return multiply(a, y); }, x, x_tan).second;
    CHECK_EQ(out.item<complex64_t>(), complex64_t{3.0, 6.0});

    auto cotan = array(complex64_t{2.0, 3.0});
    out = vjp([x](array a) { return multiply(a, x); }, y, cotan).second;
    CHECK_EQ(out.dtype(), float32);
    CHECK_EQ(out.item<float>(), -8.0);

    out = vjp([y](array a) { return multiply(a, y); }, x, cotan).second;
    CHECK_EQ(out.item<complex64_t>(), complex64_t{6.0, 9.0});
  }
}

TEST_CASE("test scan grads") {
  // Test cumsum
  {
    int axis = 0;
    int reverse = false;
    int inclusive = true;
    auto fun = [&axis, &reverse, &inclusive](array x) {
      return cumsum(x, axis, reverse, inclusive);
    };

    auto out = vjp(fun, ones({4}), ones({4})).second;
    auto expected = array({4.0f, 3.0f, 2.0f, 1.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    reverse = true;
    out = vjp(fun, ones({4}), ones({4})).second;
    expected = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    reverse = true;
    inclusive = false;
    out = vjp(fun, ones({4}), ones({4})).second;
    expected = array({0.0f, 1.0f, 2.0f, 3.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    reverse = false;
    inclusive = false;
    out = vjp(fun, ones({4}), ones({4})).second;
    expected = array({3.0f, 2.0f, 1.0f, 0.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());
  }

  // Test cumprod
  {
    int axis = 0;
    int reverse = false;
    int inclusive = true;
    auto fun = [&axis, &reverse, &inclusive](array x) {
      return cumprod(x, axis, reverse, inclusive);
    };

    auto x = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    auto g = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    auto out = vjp(fun, x, g).second;
    auto expected = array({119.0f, 59.0f, 38.0f, 24.0f}, {4});
    CHECK(allclose(out, expected).item<bool>());

    reverse = true;
    out = vjp(fun, x, g).second;
    expected = array({24.0f, 36.0f, 36.0f, 31.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    inclusive = false;
    out = vjp(fun, x, g).second;
    expected = array({0.0f, 12.0f, 16.0f, 15.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    reverse = false;
    out = vjp(fun, x, g).second;
    expected = array({32.0f, 15.0f, 8.0f, 0.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());
  }

  // Test cumsum jvp
  {
    int axis = 0;
    int reverse = false;
    int inclusive = true;
    auto fun = [&axis, &reverse, &inclusive](array x) {
      return cumsum(x, axis, reverse, inclusive);
    };

    auto x = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    auto out = jvp(fun, x, ones({4})).second;
    auto expected = array({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    reverse = true;
    out = jvp(fun, x, ones({4})).second;
    expected = array({4.0f, 3.0f, 2.0f, 1.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    inclusive = false;
    out = jvp(fun, x, ones({4})).second;
    expected = array({3.0f, 2.0f, 1.0f, 0.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());

    reverse = false;
    out = jvp(fun, x, ones({4})).second;
    expected = array({0.0f, 1.0f, 2.0f, 3.0f}, {4});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test update state") {
  auto y = array({1.0});
  auto x = array({1.0, 1.0});
  auto state = array({0.0, 0.0});
  auto fn = [&state, &x](array y) {
    x = y * x;
    state = state + x;
    return sum(x);
  };
  grad(fn)(y);
  eval(state);
  CHECK(!state.has_primitive());
  CHECK(state.is_evaled());
  CHECK(array_equal(state, array({1.0, 1.0})).item<bool>());
}

TEST_CASE("test grad types") {
  {
    auto fn = [](array x) { return sum(x); };

    for (auto t : {float16, bfloat16, float32}) {
      auto x = array(1.0, t);
      auto dfdx = grad(fn)(x);
      CHECK_EQ(dfdx.dtype(), t);
    }
  }

  {
    // Check for multi-input grad
    auto fn = [](std::vector<array> inputs) {
      return sum(inputs[0] + inputs[1]);
    };

    for (auto t : {float16, bfloat16, float32}) {
      auto x = array(1.0, t);
      auto y = array(1.0, t);
      auto out = grad(fn)({x, y});
      CHECK_EQ(out[0].dtype(), t);
    }
  }
}
