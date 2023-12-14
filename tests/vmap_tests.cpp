// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test simple vmap") {
  // vmap reshape
  {
    auto vfun = vmap([](array input) { return reshape(input, {2, 2}); });
    auto x = zeros({3, 4});
    CHECK(array_equal(vfun(x), zeros({3, 2, 2})).item<bool>());

    x = array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 2, 2});
    vfun = vmap([](array input) { return reshape(input, {4}); });
    auto expected = array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 4});
    CHECK(array_equal(vfun(x), expected).item<bool>());

    vfun = vmap([](array input) { return reshape(input, {4}); }, 1);
    expected = array({0, 1, 4, 5, 2, 3, 6, 7}, {2, 4});
    CHECK(array_equal(vfun(x), expected).item<bool>());

    vfun = vmap([](array input) { return reshape(input, {4}); }, 1, 1);
    expected = array({0, 2, 1, 3, 4, 6, 5, 7}, {4, 2});
    CHECK(array_equal(vfun(x), expected).item<bool>());
  }

  // vmap broadcast
  {
    auto fun = [](array input) { return broadcast_to(input, {4, 2}); };

    CHECK_THROWS_AS(vmap(fun, 0, -1), std::invalid_argument);
    CHECK_THROWS_AS(vmap(fun, -1, 0), std::invalid_argument);

    auto vfun = vmap(fun, -1, -1);
    auto x = zeros({2});
    CHECK(array_equal(vfun(x), zeros({4, 2})).item<bool>());

    vfun = vmap(fun);
    x = zeros({3, 2});
    CHECK(array_equal(vfun(x), zeros({3, 4, 2})).item<bool>());

    vfun = vmap(fun, 0, 1);
    CHECK(array_equal(vfun(x), zeros({4, 3, 2})).item<bool>());

    vfun = vmap(fun, 0, 2);
    CHECK(array_equal(vfun(x), zeros({4, 2, 3})).item<bool>());

    vfun = vmap(fun, 0, 2);
    x = zeros({2, 3});
    CHECK_THROWS_AS(vfun(x), std::invalid_argument);

    x = zeros({2, 3});
    vfun = vmap(fun, 1);
    CHECK(array_equal(vfun(x), zeros({3, 4, 2})).item<bool>());

    vfun = vmap(fun, 1, 1);
    CHECK(array_equal(vfun(x), zeros({4, 3, 2})).item<bool>());

    vfun = vmap(fun, 1, 2);
    CHECK(array_equal(vfun(x), zeros({4, 2, 3})).item<bool>());
  }

  // vmap transpose
  {
    auto fun = [](array input) { return transpose(input); };
    auto vfun = vmap(fun);
    auto x = array({0, 1, 2, 3, 4, 5}, {3, 2});
    CHECK(array_equal(vfun(x), x).item<bool>());

    vfun = vmap(fun, 0, 1);
    CHECK(array_equal(vfun(x), transpose(x)).item<bool>());

    x = array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 2, 2});
    vfun = vmap(fun);
    CHECK(array_equal(vfun(x), transpose(x, {0, 2, 1})).item<bool>());

    vfun = vmap(fun, 1, 1);
    CHECK(array_equal(vfun(x), transpose(x, {2, 1, 0})).item<bool>());

    vfun = vmap(fun, 2, 2);
    CHECK(array_equal(vfun(x), transpose(x, {1, 0, 2})).item<bool>());

    // vmap twice
    x = array(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {2, 2, 2, 2});
    vfun = vmap(vmap(fun));
    CHECK(array_equal(vfun(x), transpose(x, {0, 1, 3, 2})).item<bool>());
  }

  // vmap add
  {
    auto fun = [](std::vector<array> inputs) {
      auto out = add(inputs[0], inputs[1]);
      return std::vector<array>{out};
    };

    auto vfun = vmap(fun);
    array x({1.0, 2.0}, {2, 1});
    array y({2.0, 3.0}, {2, 1});
    auto out = vfun({x, y})[0];
    CHECK(array_equal(out, array({3.0, 5.0}, {2, 1})).item<bool>());

    x = ones({2, 1, 3});
    y = ones({3, 2});
    vfun = vmap(fun, {2, 0});
    out = vfun({x, y})[0];
    CHECK(array_equal(out, full({3, 2, 2}, 2.0)).item<bool>());

    x = array(1.);
    y = ones({3, 2});
    vfun = vmap(fun, {-1, 0});
    out = vfun({x, y})[0];
    CHECK(array_equal(out, full({3, 2}, 2.0)).item<bool>());

    x = ones({3, 2});
    y = array(1.);
    vfun = vmap(fun, {0, -1});
    out = vfun({x, y})[0];
    CHECK(array_equal(out, full({3, 2}, 2.0)).item<bool>());

    CHECK_THROWS_AS(vmap(fun, {-1, -1}, {0}), std::invalid_argument);
    CHECK_THROWS_AS(vmap(fun, {-1, 0}, {-1}), std::invalid_argument);
    CHECK_THROWS_AS(vmap(fun, {0, -1}, {-1}), std::invalid_argument);

    x = array(1.);
    y = array(1.);
    vfun = vmap(fun, {-1, -1}, {-1});
    out = vfun({x, y})[0];
    CHECK(array_equal(out, array(2.)).item<bool>());

    x = ones({3, 2, 1});
    y = ones({3, 2, 1});
    vfun = vmap(vmap(fun));
    out = vfun({x, y})[0];
    CHECK(array_equal(out, x + y).item<bool>());
  }

  // vmap with capturing closure
  {
    auto x = add(add(ones({2}), zeros({2})), zeros({2}));
    auto fun = [x](const array& input) { return add(input, x); };

    auto vfun = vmap(fun);
    auto y = ones({3, 2});
    CHECK(array_equal(vfun(y), full({3, 2}, 2.0f)).item<bool>());
  }
  {
    auto x = ones({4});
    auto z = x + x;
    auto vfun = vmap(
        [z](std::vector<array> inputs) {
          return std::vector<array>{add(z, inputs[1])};
        },
        {-1, 0});
    auto y = ones({3, 4});
    CHECK(array_equal(vfun({x, y})[0], full({3, 4}, 3.0)).item<bool>());
  }
}

TEST_CASE("test vmap with eval") {
  auto fun = [](std::vector<array> inputs) {
    auto x = inputs[0] + 1;
    auto y = inputs[1] + 2;
    eval(x);
    auto out = add(x, y);
    return std::vector<array>{out};
  };

  auto vfun = vmap(fun);
  array x({1.0, 2.0}, {2, 1});
  array y({2.0, 3.0}, {2, 1});
  CHECK_THROWS(vfun({x, y}));

  // Ok to eval functions of non-vmapped input
  x = array(1.0);
  vfun = vmap(fun, {-1, 0});
  CHECK(array_equal(vfun({x, y})[0], array({6.0f, 7.0f}, {2, 1})).item<bool>());

  // Not ok to eval function of vmapped input even with retain graph
  auto fun2 = [](std::vector<array> inputs) {
    auto x = inputs[0] + 1;
    auto y = inputs[1] + 2;
    eval({x}, true);
    auto out = add(x, y);
    return std::vector<array>{out};
  };
  x = array({1.0, 2.0}, {2, 1});
  CHECK_THROWS(vmap(fun2)({x, y}));
}

TEST_CASE("test vmap comparison ops") {
  // vmap equal
  {
    auto fun = [](std::vector<array> inputs) {
      return std::vector<array>{equal(inputs[0], inputs[1])};
    };
    auto vfun = vmap(fun);
    auto x = zeros({2, 3}, float32);
    auto y = zeros({2, 3}, float32);
    auto out = vfun({x, y})[0];
    CHECK(all(out).item<bool>());

    vfun = vmap(fun, {0, -1});
    x = zeros({2, 3}, float32);
    y = zeros({3}, float32);
    out = vfun({x, y})[0];
    CHECK(all(out).item<bool>());

    vfun = vmap(fun, {0, -1});
    x = array({0, 0, 0, 1, 1, 1}, {2, 3});
    y = zeros({3}, float32);
    out = vfun({x, y})[0];
    auto expected = array({true, true, true, false, false, false}, {2, 3});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test vmap creation ops") {
  // vmap astype
  {
    auto fun = [](array in) { return astype(in, int32); };
    auto x = zeros({2, 3}, float32);
    auto out = vmap(fun)(x);
    CHECK_EQ(out.dtype(), int32);
    CHECK(array_equal(out, zeros({2, 3}, int32)).item<bool>());
  }

  // vmap full
  {
    auto fun = [](array in) { return full({2}, in); };
    auto x = array({1, 2, 3});
    auto out = vmap(fun)(x);
    auto expected = array({1, 1, 2, 2, 3, 3}, {3, 2});
    CHECK(array_equal(out, expected).item<bool>());

    x = array({1, 2, 3}, {3, 1});
    out = vmap(fun)(x);
    expected = array({1, 1, 2, 2, 3, 3}, {3, 2});
    CHECK(array_equal(out, expected).item<bool>());

    x = array({1, 2, 3}, {1, 3});
    CHECK_THROWS_AS(vmap(fun)(x), std::invalid_argument);
    out = vmap(fun, 1, 1)(x);
    expected = array({1, 2, 3, 1, 2, 3}, {2, 3});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test vmap slice") {
  {
    auto fun = [](array in) { return slice(in, {4}, {8}, {2}); };
    auto x = reshape(arange(16), {2, 8});
    auto out = vmap(fun)(x);
    auto expected = reshape(array({4, 6, 12, 14}), {2, 2});
    CHECK(array_equal(out, expected).item<bool>());
  }

  {
    auto fun = [](array in) { return slice(in, {0, 1}, {2, 3}); };
    auto x = reshape(arange(12), {2, 2, 3});
    auto out = vmap(fun, 1, 0)(x);
    auto expected = reshape(array({1, 2, 7, 8, 4, 5, 10, 11}), {2, 2, 2});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test vmap concatenate") {
  auto fun = [](std::vector<array> inputs) {
    return std::vector<array>{concatenate(inputs, 0)};
  };
  auto x = reshape(arange(4), {2, 2});
  auto y = reshape(arange(4), {2, 2});
  auto out = vmap(fun)({x, y})[0];
  auto expected = reshape(array({0, 1, 0, 1, 2, 3, 2, 3}), {2, 4});
  CHECK(array_equal(out, expected).item<bool>());
  out = vmap(fun, {1, 1})({x, y})[0];
  expected = reshape(array({0, 2, 0, 2, 1, 3, 1, 3}), {2, 4});
  CHECK(array_equal(out, expected).item<bool>());
  out = vmap(fun, {0, 1})({x, y})[0];
  expected = reshape(array({0, 1, 0, 2, 2, 3, 1, 3}), {2, 4});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test vmap gather") {
  {
    auto fun = [](std::vector<array> inputs) {
      auto src = inputs[0];
      auto indices = inputs[1];
      std::vector<int> slice_sizes = {1, 2, 2};
      auto out = squeeze(gather(src, indices, 0, slice_sizes), 2);
      return std::vector<array>{out};
    };
    auto x = zeros({2, 2, 2, 2});
    auto y = array({0, 1, 0, 0, 1, 0}, {2, 3});
    auto out = vmap(fun, {0, -1})({x, y})[0];
    CHECK_EQ(out.shape(), std::vector<int>{2, 2, 3, 2, 2});
    out = vmap(fun, {0, -1}, {3})({x, y})[0];
    CHECK_EQ(out.shape(), std::vector<int>{2, 3, 2, 2, 2});
  }

  {
    auto fun = [](std::vector<array> inputs) {
      auto src = inputs[0];
      auto indices = inputs[1];
      std::vector<int> slice_sizes = {1, 2, 2};
      auto out = squeeze(gather(src, indices, 0, slice_sizes), 1);
      return std::vector<array>{out};
    };
    auto x = zeros({2, 2, 2, 2});
    auto y = array({0, 1, 0, 0, 1, 0}, {2, 3});
    auto out = vmap(fun, {0, 0})({x, y})[0];
    CHECK_EQ(out.shape(), std::vector<int>{2, 3, 2, 2});
  }

  {
    auto fun = [](std::vector<array> inputs) {
      auto src = inputs[0];
      auto indices = inputs[1];
      std::vector<int> slice_sizes = {1, 2, 2, 2};
      auto out = squeeze(gather(src, indices, 0, slice_sizes), 1);
      return std::vector<array>{out};
    };
    auto x = zeros({2, 2, 2, 2});
    auto y = array({0, 1, 0, 0, 1, 0}, {2, 3});

    auto out = vmap(fun, {-1, 0})({x, y})[0];
    CHECK_EQ(out.shape(), std::vector<int>{2, 3, 2, 2, 2});
  }

  {
    auto fun = [](std::vector<array> inputs) {
      auto src = inputs[0];
      auto indices = std::vector<array>(inputs.begin() + 1, inputs.end());
      std::vector<int> slice_sizes = {1, 1, 2, 2};
      auto out = squeeze(gather(src, indices, {0, 1}, slice_sizes), {1, 2});
      return std::vector<array>{out};
    };
    auto x = zeros({2, 2, 2, 2});
    auto y = array({0, 1, 0, 0, 1, 0}, {2, 3});
    auto z = array({0, 1, 0, 0, 1, 0}, {2, 3});
    auto out = vmap(fun, {-1, 0, 0})({x, y, z})[0];
    CHECK_EQ(out.shape(), std::vector<int>{2, 3, 2, 2});

    z = array({0, 1, 0, 0, 1, 0}, {3, 2});
    out = vmap(fun, {-1, 0, 1})({x, y, z})[0];
    CHECK_EQ(out.shape(), std::vector<int>{2, 3, 2, 2});
  }
}
