// Copyright © 2023-2024 Apple Inc.

// Required for using M_PI_2 in MSVC.
#define _USE_MATH_DEFINES

#include <cmath>
#include <numeric>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test copy") {
  array x(1.0);
  auto y = copy(x);
  CHECK_EQ(y.shape(), Shape{});
  CHECK_NE(y.id(), x.id());
  CHECK_EQ(y.item<float>(), 1.0f);

  x = array({1, 2}, {2, 1});
  y = copy(x);
  CHECK_EQ(y.shape(), Shape{2, 1});
  CHECK_EQ(y.dtype(), int32);
  CHECK_NE(y.id(), x.id());
  CHECK(array_equal(y, x).item<bool>());
}

TEST_CASE("test reshape") {
  array x(1.0);
  CHECK_EQ(reshape(x, {}).shape(), Shape{});
  CHECK_THROWS_AS(reshape(x, {2}), std::invalid_argument);
  auto y = reshape(x, {1, 1, 1});
  CHECK_EQ(y.shape(), Shape{1, 1, 1});
  y = reshape(x, {-1, 1, 1});
  CHECK_EQ(y.shape(), Shape{1, 1, 1});
  y = reshape(x, {1, 1, -1});
  CHECK_EQ(y.shape(), Shape{1, 1, 1});
  CHECK_THROWS_AS(reshape(x, {1, -1, -1}), std::invalid_argument);
  CHECK_THROWS_AS(reshape(x, {2, -1}), std::invalid_argument);

  x = zeros({2, 2, 2});
  y = reshape(x, {8});
  CHECK_EQ(y.shape(), Shape{8});
  CHECK_THROWS_AS(reshape(x, {7}), std::invalid_argument);
  y = reshape(x, {-1});
  CHECK_EQ(y.shape(), Shape{8});
  y = reshape(x, {-1, 2});
  CHECK_EQ(y.shape(), Shape{4, 2});
  CHECK_THROWS_AS(reshape(x, {-1, 7}), std::invalid_argument);

  // Works with empty array
  x = array({});
  y = reshape(x, {0, 0, 0});
  CHECK_EQ(y.shape(), Shape{0, 0, 0});
  y.eval();
  CHECK_EQ(y.size(), 0);
  CHECK_THROWS_AS(reshape(x, {}), std::invalid_argument);
  CHECK_THROWS_AS(reshape(x, {1}), std::invalid_argument);
  y = reshape(x, {1, 5, 0});
  CHECK_EQ(y.shape(), Shape{1, 5, 0});

  // Check that reshaping a transposed array doesn't result in a copy
  x = reshape(arange(64), {2, 4, 8});
  x.eval();
  CHECK_EQ(x.strides()[0], 32);
  CHECK_EQ(x.strides()[1], 8);
  CHECK_EQ(x.strides()[2], 1);
  y = reshape(transpose(x, {0, 2, 1}), {2, 4, 2, 4});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 2);
  CHECK_EQ(y.strides()[2], 1);
  CHECK_EQ(y.strides()[3], 8);
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Split transposed (2, 8, 4) -> (2, 8, 2, 2)
  y = reshape(transpose(x, {0, 2, 1}), {2, 8, 2, 2});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 1);
  CHECK_EQ(y.strides()[2], 16);
  CHECK_EQ(y.strides()[3], 8);
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Split transposed (2, 8, 4) -> (2, 8, 2, 1, 2)
  y = reshape(transpose(x, {0, 2, 1}), {2, 8, 2, 1, 2});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 1);
  CHECK_EQ(y.strides()[2], 16);
  // y.strides()[3] can be anything since y.shape()[3] == 1
  CHECK_EQ(y.strides()[4], 8);
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Split transposed (2, 8, 4) -> (2, 8, 2, 1, 2, 1)
  y = reshape(transpose(x, {0, 2, 1}), {2, 8, 2, 1, 2, 1});
  y.eval();
  CHECK_EQ(y.strides()[0], 32);
  CHECK_EQ(y.strides()[1], 1);
  CHECK_EQ(y.strides()[2], 16);
  // y.strides()[3] can be anything since y.shape()[3] == 1
  CHECK_EQ(y.strides()[4], 8);
  // y.strides()[5] can be anything since y.shape()[5] == 1
  CHECK_EQ(x.data<int32_t>(), y.data<int32_t>());

  // Check contiguity preservation
  x = ones({10, 10});
  eval(x);
  CHECK(x.flags().row_contiguous);
  CHECK(!x.flags().col_contiguous);
  y = reshape(x, {2, 5, 10});
  eval(y);
  CHECK(y.flags().row_contiguous);
  CHECK(!y.flags().col_contiguous);
  y = reshape(x, {10, 1, 10, 1});
  eval(y);
  CHECK(y.flags().row_contiguous);
  CHECK(!y.flags().col_contiguous);
  x = transpose(x, {1, 0});
  eval(x);
  CHECK(!x.flags().row_contiguous);
  CHECK(x.flags().col_contiguous);
  y = reshape(x, {2, 5, 10});
  eval(y);
  CHECK(!y.flags().row_contiguous);
  CHECK(y.flags().col_contiguous);
  y = reshape(x, {2, 50});
  eval(y);
  CHECK(y.flags().row_contiguous);
  CHECK(!y.flags().col_contiguous);
  y = reshape(x, {10, 1, 10, 1});
  eval(y);
  CHECK(!y.flags().row_contiguous);
  CHECK(y.flags().col_contiguous);
}

TEST_CASE("test flatten") {
  array x = zeros({2, 3, 4});
  CHECK_EQ(flatten(x).shape(), Shape({2 * 3 * 4}));

  CHECK_EQ(flatten(x, 1, 1).shape(), Shape({2, 3, 4}));
  CHECK_EQ(flatten(x, 1, 2).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, 1, 3).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, 1, -1).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, -2, -1).shape(), Shape({2, 3 * 4}));
  CHECK_EQ(flatten(x, -3, -1).shape(), Shape({2 * 3 * 4}));
  CHECK_EQ(flatten(x, -4, -1).shape(), Shape({2 * 3 * 4}));

  // Check start > end throws
  CHECK_THROWS(flatten(x, 2, 1));

  // Check start >= ndim throws
  CHECK_THROWS(flatten(x, 5, 6));

  // Check end < 0 throws
  CHECK_THROWS(flatten(x, -5, -4));

  // Check scalar flattens to 1D
  x = array(1);
  CHECK_EQ(flatten(x, -3, -1).shape(), Shape({1}));
  CHECK_EQ(flatten(x, 0, 0).shape(), Shape({1}));
}

TEST_CASE("test unflatten") {
  array x = array(1);
  CHECK_THROWS(unflatten(x, 0, {1, 1}));

  x = array({1});
  auto out = unflatten(x, 0, {1, 1});
  CHECK_EQ(out.shape(), Shape({1, 1}));
  CHECK_THROWS(unflatten(x, 1, {1, 1}));
  CHECK_THROWS(unflatten(x, 0, {-1, -1}));
  CHECK_THROWS(unflatten(x, 0, {-1, 2}));
  CHECK_THROWS(unflatten(x, 0, {}));

  x = zeros({4, 8});
  out = unflatten(x, 1, {2, 2, 2});
  CHECK_EQ(out.shape(), Shape({4, 2, 2, 2}));
}

TEST_CASE("test squeeze and expand") {
  array x = zeros({2, 1, 2, 1, 2, 1});
  CHECK_EQ(squeeze(x).shape(), Shape{2, 2, 2});
  CHECK_EQ(squeeze(x, {1, 3, 5}).shape(), Shape{2, 2, 2});
  CHECK_EQ(squeeze(x, {-1, -3, -5}).shape(), Shape{2, 2, 2});
  CHECK_EQ(squeeze(x, 1).shape(), Shape{2, 2, 1, 2, 1});
  CHECK_EQ(squeeze(x, -1).shape(), Shape{2, 1, 2, 1, 2});

  CHECK_THROWS(squeeze(x, 0));
  CHECK_THROWS(squeeze(x, 2));
  CHECK_THROWS(squeeze(x, {1, 3, 1}));
  CHECK_THROWS(squeeze(x, {1, 3, -3}));

  x = zeros({2, 2});
  CHECK_EQ(expand_dims(x, 0).shape(), Shape{1, 2, 2});
  CHECK_EQ(expand_dims(x, -1).shape(), Shape{2, 2, 1});
  CHECK_EQ(expand_dims(x, 1).shape(), Shape{2, 1, 2});
  CHECK_EQ(expand_dims(x, {0, 1, 2}).shape(), Shape{1, 1, 1, 2, 2});
  CHECK_EQ(
      expand_dims(x, {0, 1, 2, 5, 6, 7}).shape(),
      Shape{1, 1, 1, 2, 2, 1, 1, 1});

  CHECK_THROWS(expand_dims(x, 3));
  CHECK_THROWS(expand_dims(x, -4));
  CHECK_THROWS(expand_dims(x, {0, 1, 0}));
  CHECK_THROWS(expand_dims(x, {0, 1, -4}));
}

TEST_CASE("test slice") {
  array x = array(3);
  auto out = slice(x, {}, {});
  CHECK_EQ(out.item<int>(), 3);
  CHECK_THROWS_AS(slice(x, {1}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(slice(x, {}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(slice(x, {0}, {}), std::invalid_argument);

  x = array({3});
  out = slice(x, {0}, {1});
  CHECK_EQ(out.item<int>(), 3);
  out = slice(x, {-1}, {1});
  CHECK_EQ(out.item<int>(), 3);

  out = slice(x, {-3}, {10});
  CHECK_EQ(out.item<int>(), 3);

  out = slice(x, {1}, {0});
  eval(out);
  CHECK_EQ(out.shape(), Shape{0});

  out = slice(x, {0}, {1}, {1});
  CHECK_EQ(out.item<int>(), 3);

  out = slice(x, {0}, {1}, {10});
  CHECK_EQ(out.item<int>(), 3);

  x = array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 4});
  out = slice(x, {0, 0}, {2, 2});
  CHECK(array_equal(out, array({0, 1, 4, 5}, {2, 2})).item<bool>());

  out = slice(x, {0, 0}, {0, 2});
  CHECK(array_equal(out, reshape(array({}), {0, 2})).item<bool>());

  out = slice(x, {0, 2}, {2, 3});
  CHECK(array_equal(out, array({2, 6}, {2, 1})).item<bool>());

  out = slice(x, {0, 0}, {2, 4}, {1, 2});
  CHECK(array_equal(out, array({0, 2, 4, 6}, {2, 2})).item<bool>());

  // Check contiguity preservation
  x = ones({10, 10});
  eval(x);
  CHECK(x.flags().row_contiguous);
  CHECK(!x.flags().col_contiguous);
  out = slice(x, {0, 0}, {10, 5});
  eval(out);
  CHECK(!out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);
  out = slice(x, {0, 0}, {5, 10});
  eval(out);
  CHECK(out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);
  x = transpose(x, {1, 0});
  eval(x);
  CHECK(!x.flags().row_contiguous);
  CHECK(x.flags().col_contiguous);
  out = slice(x, {0, 0}, {10, 5});
  eval(out);
  CHECK(!out.flags().row_contiguous);
  CHECK(out.flags().col_contiguous);
  out = slice(x, {0, 0}, {5, 10});
  eval(out);
  CHECK(!out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);

  x = ones({6, 4, 10});
  out = slice(x, {0, 0, 0}, {6, 4, 10}, {2, 1, 2});
  eval(out);
  CHECK(!out.flags().contiguous);
  CHECK(!out.flags().row_contiguous);
  CHECK(!out.flags().col_contiguous);

  // Check data size correctness
  x = ones({4});
  out = slice(x, {0}, {2});
  eval(out);
  CHECK_EQ(out.data_size(), 2);

  out = slice(x, {2}, {4});
  eval(out);
  CHECK_EQ(out.data_size(), 2);

  out = slice(x, {0}, {4}, {2});
  eval(out);
  CHECK_EQ(out.data_size(), 4);

  x = ones({4, 4});
  out = slice(x, {0, 0}, {2, 4});
  eval(out);
  CHECK_EQ(out.data_size(), 8);

  out = slice(x, {0, 0}, {1, 2});
  eval(out);
  CHECK_EQ(out.data_size(), 2);

  out = slice(x, {0, 1}, {4, 4});
  eval(out);
  CHECK_EQ(out.data_size(), 15);

  out = slice(x, {1, 2}, {3, 4});
  eval(out);
  CHECK_EQ(out.data_size(), 6);

  x = ones({4, 4, 4});
  out = slice(x, {0, 0, 0}, {4, 2, 2});
  eval(out);
  CHECK_EQ(out.data_size(), 54);

  x = ones({4, 4, 4});
  out = slice(x, {2, 2, 2}, {3, 3, 3});
  eval(out);
  CHECK_EQ(out.data_size(), 1);

  x = ones({4, 4, 4});
  out = slice(x, {2, 2, 2}, {3, 4, 3});
  eval(out);
  CHECK_EQ(out.data_size(), 5);
}

TEST_CASE("test slice update") {
  array x = array({0., 0., 0., 0., 0., 0., 0., 0.}, {8}, float32);
  array y = array(
      {
          1.,
          2.,
          3.,
          4.,
      },
      {4},
      float32);

  auto out = slice_update(x, y, {2}, {6}, {1});
  CHECK(array_equal(slice(out, {2}, {6}, {1}), y).item<bool>());

  out = slice_update(x, y, {5}, {1}, {-1});
  CHECK(array_equal(slice(out, {5}, {1}, {-1}), y).item<bool>());

  x = reshape(x, {2, 4});
  out = slice_update(x, y, {0, 0}, {2, 4}, {1, 1});
  out = reshape(out, {8});
  CHECK(array_equal(slice(out, {0}, {4}, {1}), y).item<bool>());
  CHECK(array_equal(slice(out, {4}, {8}, {1}), y).item<bool>());
}

TEST_CASE("test split") {
  array x = array(1);
  CHECK_THROWS(split(x, 0));

  x = array({3});
  CHECK_EQ(split(x, 1)[0].item<int>(), 3);

  x = array({0, 1, 2});
  CHECK_THROWS(split(x, 3, 1));
  CHECK_THROWS(split(x, 3, -2));

  auto out = split(x, 3, 0);
  CHECK_EQ(out.size(), 3);

  out = split(x, 3, -1);
  CHECK_EQ(out.size(), 3);
  for (auto i = 0; i < 3; ++i) {
    CHECK_EQ(out[i].shape(), Shape{1});
    CHECK_EQ(out[i].dtype(), int32);
    CHECK_EQ(out[i].item<int>(), i);
  }

  x = array({0, 1, 2, 3, 4, 5}, {2, 3});
  out = split(x, 2);
  CHECK(array_equal(out[0], array({0, 1, 2}, {1, 3})).item<bool>());
  CHECK(array_equal(out[1], array({3, 4, 5}, {1, 3})).item<bool>());
  out = split(x, 3, 1);
  CHECK(array_equal(out[0], array({0, 3}, {2, 1})).item<bool>());
  CHECK(array_equal(out[1], array({1, 4}, {2, 1})).item<bool>());
  CHECK(array_equal(out[2], array({2, 5}, {2, 1})).item<bool>());

  x = zeros({8, 12});
  out = split(x, 2);
  CHECK_EQ(out.size(), 2);
  CHECK_EQ(out[0].shape(), Shape{4, 12});
  CHECK_EQ(out[1].shape(), Shape{4, 12});
  out = split(x, 3, 1);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].shape(), Shape{8, 4});
  CHECK_EQ(out[1].shape(), Shape{8, 4});
  CHECK_EQ(out[2].shape(), Shape{8, 4});

  out = split(x, Shape{});
  CHECK_EQ(out.size(), 1);
  CHECK_EQ(out[0].shape(), x.shape());

  out = split(x, {3, 7});
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].shape(), Shape{3, 12});
  CHECK_EQ(out[1].shape(), Shape{4, 12});
  CHECK_EQ(out[2].shape(), Shape{1, 12});

  out = split(x, Shape{20});
  CHECK_EQ(out.size(), 2);
  CHECK_EQ(out[0].shape(), Shape{8, 12});
  CHECK_EQ(out[1].shape(), Shape{0, 12});

  // Negative indices
  out = split(x, Shape{-5});
  CHECK_EQ(out[0].shape(), Shape{3, 12});
  CHECK_EQ(out[1].shape(), Shape{5, 12});

  // Different axis
  out = split(x, {2, 8}, 1);
  CHECK_EQ(out[0].shape(), Shape{8, 2});
  CHECK_EQ(out[1].shape(), Shape{8, 6});
  CHECK_EQ(out[2].shape(), Shape{8, 4});

  // Out of order indices
  x = arange(5);
  out = split(x, {2, 1, 2});
  CHECK(array_equal(out[0], array({0, 1})).item<bool>());
  CHECK(array_equal(out[1], array({})).item<bool>());
  CHECK(array_equal(out[2], array({1})).item<bool>());
  CHECK(array_equal(out[3], array({2, 3, 4})).item<bool>());
}

TEST_CASE("test swap and move axes") {
  // Test swapaxes
  array a(0.0);
  CHECK_THROWS(swapaxes(a, 0, 0));

  a = zeros({2});
  CHECK_THROWS(swapaxes(a, 0, 1));
  CHECK_EQ(swapaxes(a, 0, 0).shape(), Shape{2});
  CHECK_EQ(swapaxes(a, -1, -1).shape(), Shape{2});

  a = zeros({2, 3, 4});
  CHECK_THROWS(swapaxes(a, 0, -4));
  CHECK_THROWS(swapaxes(a, 0, 3));
  CHECK_THROWS(swapaxes(a, 3, 0));
  CHECK_THROWS(swapaxes(a, -4, 0));
  CHECK_EQ(swapaxes(a, 0, 2).shape(), Shape{4, 3, 2});
  CHECK_EQ(swapaxes(a, 0, 1).shape(), Shape{3, 2, 4});
  CHECK_EQ(swapaxes(a, 0, -1).shape(), Shape{4, 3, 2});
  CHECK_EQ(swapaxes(a, -2, 2).shape(), Shape{2, 4, 3});

  // Test moveaxis
  a = array(0.0);
  CHECK_THROWS(moveaxis(a, 0, 0));

  a = zeros({2});
  CHECK_THROWS(moveaxis(a, 0, 1));
  CHECK_EQ(moveaxis(a, 0, 0).shape(), Shape{2});
  CHECK_EQ(moveaxis(a, -1, -1).shape(), Shape{2});

  a = zeros({2, 3, 4});
  CHECK_THROWS(moveaxis(a, 0, -4));
  CHECK_THROWS(moveaxis(a, 0, 3));
  CHECK_THROWS(moveaxis(a, 3, 0));
  CHECK_THROWS(moveaxis(a, -4, 0));
  CHECK_EQ(moveaxis(a, 0, 2).shape(), Shape{3, 4, 2});
  CHECK_EQ(moveaxis(a, 0, 1).shape(), Shape{3, 2, 4});
  CHECK_EQ(moveaxis(a, 0, -1).shape(), Shape{3, 4, 2});
  CHECK_EQ(moveaxis(a, -2, 2).shape(), Shape{2, 4, 3});
}

TEST_CASE("test transpose") {
  array x(1);
  auto y = transpose(x);
  CHECK_EQ(y.shape(), Shape{});
  CHECK_EQ(y.item<int>(), 1);
  CHECK_THROWS_AS(transpose(x, {0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {1}), std::invalid_argument);

  x = array({1}, {1});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{1});
  CHECK_EQ(y.item<int>(), 1);

  // Negative indices
  y = transpose(x, {-1});
  CHECK_EQ(y.shape(), Shape{1});
  CHECK_EQ(y.item<int>(), 1);

  CHECK_THROWS_AS(transpose(x, {1}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0, 0}), std::invalid_argument);

  // Works with empty array
  x = array({});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{0});
  y.eval();
  CHECK_EQ(y.size(), 0);

  x = array({1, 2, 3, 4, 5, 6}, {2, 3});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{3, 2});
  y = transpose(x, {-1, 0});
  CHECK_EQ(y.shape(), Shape{3, 2});
  y = transpose(x, {-1, -2});
  CHECK_EQ(y.shape(), Shape{3, 2});
  y.eval();
  CHECK(array_equal(y, array({1, 4, 2, 5, 3, 6}, {3, 2})).item<bool>());
  y = transpose(x, {0, 1});
  CHECK_EQ(y.shape(), Shape{2, 3});
  CHECK(array_equal(y, x).item<bool>());
  y = transpose(x, {0, -1});
  CHECK_EQ(y.shape(), Shape{2, 3});
  CHECK(array_equal(y, x).item<bool>());

  CHECK_THROWS_AS(transpose(x, {}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0, 0, 0}), std::invalid_argument);
  CHECK_THROWS_AS(transpose(x, {0, 1, 1}), std::invalid_argument);

  x = array({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {2, 3, 2});
  y = transpose(x);
  CHECK_EQ(y.shape(), Shape{2, 3, 2});
  auto expected = array({1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12}, {2, 3, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = transpose(x, {0, 1, 2});
  CHECK_EQ(y.shape(), Shape{2, 3, 2});
  CHECK(array_equal(y, x).item<bool>());
  y = transpose(x, {1, 0, 2});
  CHECK_EQ(y.shape(), Shape{3, 2, 2});
  expected = array({1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12}, {3, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());
  y = transpose(x, {0, 2, 1});
  CHECK_EQ(y.shape(), Shape{2, 2, 3});
  expected = array({1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12}, {2, 2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  // Check reshaping a transposed array
  x = array({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2});
  x = reshape(transpose(x), {2, 2, 2});
  expected = array({0, 2, 4, 6, 1, 3, 5, 7}, {2, 2, 2});
  CHECK(array_equal(x, expected).item<bool>());

  // Check maintaining contiguous status
  x = array({0, 1, 2, 3, 4, 5, 6, 7}, {1, 4, 1, 2});
  CHECK(x.flags().row_contiguous);
  x = transpose(x, {2, 1, 0, 3});
  eval(x);
  CHECK(x.flags().row_contiguous);
}

TEST_CASE("test comparison ops") {
  // Empty array
  {
    array x({});
    array y({});
    auto z = x == y;
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.shape(), Shape{0});
  }

  // Basic cases
  {
    array x(1.0);
    array y(1.0);
    CHECK(equal(x, y).item<bool>());
    CHECK((x == y).item<bool>());
    CHECK((x == 1.0f).item<bool>());
    CHECK((1.0f == y).item<bool>());

    CHECK(!(x != y).item<bool>());
    CHECK(!not_equal(x, y).item<bool>());
    CHECK(!(1.0f != y).item<bool>());
    CHECK(!(x != 1.0f).item<bool>());

    CHECK(array_equal(x, y).item<bool>());

    x = array(0.0);
    CHECK(!equal(x, y).item<bool>());
    CHECK(!array_equal(x, y).item<bool>());
    CHECK(not_equal(x, y).item<bool>());
  }

  // Greater and less
  {
    array x(1.0);
    array y(0.0);
    CHECK(greater(x, y).item<bool>());
    CHECK((x > 0.0f).item<bool>());
    CHECK((1.0f > y).item<bool>());
    CHECK(greater_equal(x, y).item<bool>());
    CHECK((1.0f >= y).item<bool>());
    CHECK(!(x > 1.0f).item<bool>());
    CHECK((x >= 1.0f).item<bool>());

    CHECK(less(y, x).item<bool>());
    CHECK((y < 1.0).item<bool>());
    CHECK((y <= 1.0f).item<bool>());
    CHECK(!(x < 1.0).item<bool>());
    CHECK((x <= 1.0f).item<bool>());
  }

  // Check array_equal works
  {
    auto x = zeros({5, 5});
    auto y = zeros({5, 5});
    CHECK(array_equal(x, y).item<bool>());

    x = zeros({1, 1});
    CHECK(!array_equal(x, y).item<bool>());

    x = ones({5, 5});
    CHECK(!array_equal(x, y).item<bool>());

    x = array({0.0f, 1.0f, NAN});
    y = array({0.0f, 1.0f, NAN});
    CHECK(!array_equal(x, y).item<bool>());
    CHECK(array_equal(x, y, true).item<bool>());
  }

  // Check other types
  {
    auto x = zeros({5, 5}, int32);
    auto y = zeros({5, 5}, int32);
    CHECK(array_equal(x, y).item<bool>());

    x = ones({5, 5}, bool_);
    y = ones({5, 5}, bool_);
    CHECK(array_equal(x, y).item<bool>());
  }

  // Check type promotion
  {
    array x(1.0f);
    array y(1);
    CHECK_EQ(equal(x, y).item<bool>(), true);

    x = array(true, bool_);
    CHECK_EQ(equal(x, y).item<bool>(), true);
  }

  // Broadcasting works
  {
    auto x = zeros({1, 2});
    auto y = zeros({2, 1});
    auto z = equal(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.shape(), Shape{2, 2});
    auto expected = array({true, true, true, true}, {2, 2});
    CHECK(array_equal(z, expected).item<bool>());

    x = array({1.0, 2.0}, {1, 2});
    y = array({1.0, 2.0}, {2, 1});
    z = equal(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.shape(), Shape{2, 2});
    expected = array({true, false, false, true}, {2, 2});
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({false, true, false, false}, {2, 2});
    z = greater(x, y);
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({true, true, false, true}, {2, 2});
    z = greater_equal(x, y);
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({false, false, true, false}, {2, 2});
    z = less(x, y);
    CHECK(array_equal(z, expected).item<bool>());

    expected = array({true, false, true, true}, {2, 2});
    z = less_equal(x, y);
    CHECK(array_equal(z, expected).item<bool>());
  }
}

TEST_CASE("test is nan") {
  array x(1.0f);
  CHECK_FALSE(isnan(x).item<bool>());

  array y(NAN);
  CHECK(isnan(y).item<bool>());

  array z = identity(7);
  CHECK_FALSE(all(isnan(z)).item<bool>());

  array w = array({1.0f, NAN, 2.0f});
  CHECK_FALSE(all(isnan(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isnan(a).item<bool>());

  array b(1.0f, float16);
  CHECK_FALSE(isnan(b).item<bool>());

  array c(NAN, bfloat16);
  CHECK(isnan(c).item<bool>());

  array d(NAN, float16);
  CHECK(isnan(d).item<bool>());
}

TEST_CASE("test is inf") {
  array x(1.0f);
  CHECK_FALSE(isinf(x).item<bool>());

  auto inf = std::numeric_limits<float>::infinity();

  array y(inf);
  CHECK(isinf(y).item<bool>());

  auto neginf = -std::numeric_limits<float>::infinity();
  CHECK(isinf(array(neginf)).item<bool>());

  array z = identity(7);
  CHECK_FALSE(any(isinf(z)).item<bool>());

  array w = array({1.0f, inf, 2.0f});
  CHECK(array_equal(array({false, true, false}), isinf(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isinf(a).item<bool>());

  array b(1.0f, float16);
  CHECK_FALSE(isinf(b).item<bool>());

  array c(inf, bfloat16);
  CHECK(isinf(c).item<bool>());

  array d(inf, float16);
  CHECK(isinf(d).item<bool>());
}

TEST_CASE("test all close") {
  array x(1.0f);
  array y(1.0f);
  CHECK(allclose(x, y).item<bool>());

  y = array(1.1f);
  CHECK_FALSE(allclose(x, y).item<bool>());
  CHECK(allclose(x, y, 0.1).item<bool>());
  CHECK_FALSE(allclose(x, y, 0.01).item<bool>());
  CHECK(allclose(x, y, 0.01, 0.1).item<bool>());
}

TEST_CASE("test is close") {
  {
    array a({1.0, std::numeric_limits<float>::infinity()});
    array b({1.0, std::numeric_limits<float>::infinity()});
    CHECK(array_equal(isclose(a, b), array({true, true})).item<bool>());
  }
  {
    array a({1.0, -std::numeric_limits<float>::infinity()});
    array b({1.0, -std::numeric_limits<float>::infinity()});
    CHECK(array_equal(isclose(a, b), array({true, true})).item<bool>());
  }
  {
    array a({1.0, std::numeric_limits<float>::infinity()});
    array b({1.0, -std::numeric_limits<float>::infinity()});
    CHECK(array_equal(isclose(a, b), array({true, false})).item<bool>());
  }
  {
    array a({1.0, std::nan("1"), std::nan("1")});
    array b({1.0, std::nan("1"), 2.0});
    CHECK(array_equal(isclose(a, b), array({true, false, false})).item<bool>());
  }
  {
    array a({1.0, std::nan("1"), std::nan("1")});
    array b({1.0, std::nan("1"), 2.0});
    CHECK(
        array_equal(isclose(a, b, 1e-5, 1e-8, true), array({true, true, false}))
            .item<bool>());
  }
}

TEST_CASE("test reduction ops") {
  // Check shapes and throws correctly
  {
    auto x = array(1);
    auto out = sum(x);
    CHECK_EQ(out.ndim(), 0);
    CHECK_THROWS_AS(sum(x, 0), std::out_of_range);
    CHECK_THROWS_AS(sum(x, -1), std::out_of_range);
    out = sum(x, std::vector<int>{});
    CHECK_EQ(out.shape(), Shape{});
    CHECK_EQ(out.size(), 1);

    x = array({});
    out = sum(x);
    CHECK_EQ(out.shape(), Shape{});
    CHECK_EQ(out.size(), 1);
    out = sum(x, true);
    CHECK_EQ(out.shape(), Shape{1});
    out = sum(x, std::vector<int>{});
    CHECK_EQ(out.shape(), x.shape());

    x = zeros({2});
    out = sum(x);
    CHECK_EQ(out.ndim(), 0);
    out = sum(x, -1);
    CHECK_EQ(out.ndim(), 0);
    out = sum(x, -1, true);
    CHECK_EQ(out.ndim(), 1);
    CHECK_EQ(out.shape(), Shape{1});

    CHECK_THROWS_AS(sum(x, 1), std::out_of_range);
    CHECK_THROWS_AS(sum(x, -2), std::out_of_range);
    CHECK_THROWS_AS(sum(x, {0, 0}), std::invalid_argument);
    CHECK_THROWS_AS(sum(x, {-1, 0}), std::invalid_argument);

    x = zeros({2, 3, 4});
    out = sum(x, {0, 2});
    CHECK_EQ(out.shape(), Shape{3});
    out = sum(x, std::vector<int>{});
    CHECK_EQ(out.shape(), x.shape());

    out = sum(x, {0, -1});
    CHECK_EQ(out.shape(), Shape{3});

    out = sum(x, {0, -1}, true);
    CHECK_EQ(out.shape(), Shape{1, 3, 1});

    out = sum(x, true);
    CHECK_EQ(out.shape(), Shape{1, 1, 1});

    out = sum(x);
    CHECK_EQ(out.shape(), Shape{});

    CHECK_THROWS_AS(sum(x, 3), std::out_of_range);
    CHECK_THROWS_AS(sum(x, -4), std::out_of_range);
    CHECK_THROWS_AS(sum(x, {0, 1, -2}), std::invalid_argument);
  }

  // Test sum
  {
    auto x = array({});
    CHECK_EQ(sum(x).item<float>(), 0.0f);

    x = array({1, 2, 3});
    CHECK_EQ(sum(x).item<int>(), 6);
    CHECK(array_equal(sum(x, std::vector<int>{}), x).item<bool>());

    x = ones({2, 3});
    CHECK(array_equal(sum(x, 1), full({2}, 3.0f)).item<bool>());
    CHECK(array_equal(sum(x, 0), full({3}, 2.0f)).item<bool>());
    CHECK_EQ(sum(x, {0, 1}).item<float>(), 6.0f);

    x = ones({2, 3, 4});
    CHECK(array_equal(sum(x, 0), full({3, 4}, 2.0f)).item<bool>());
    CHECK(array_equal(sum(x, 1), full({2, 4}, 3.0f)).item<bool>());
    CHECK(array_equal(sum(x, 2), full({2, 3}, 4.0f)).item<bool>());
    CHECK(array_equal(sum(x, {0, 1}), full({4}, 6.0f)).item<bool>());
    CHECK(array_equal(sum(x, {0, 2}), full({3}, 8.0f)).item<bool>());
    CHECK(array_equal(sum(x, {1, 2}), full({2}, 12.0f)).item<bool>());

    // Output for bool gets higher precision
    x = array({true, true, true});
    CHECK_EQ(sum(x).item<int32_t>(), 3);

    x = array(2.0f);
    x = broadcast_to(x, {2, 2, 2});
    CHECK_EQ(sum(x).item<float>(), 16.0f);

    // Tests with non-uniform results after reduction
    x = array({1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}, {2, 3});
    CHECK(array_equal(sum(x, 0), full({3}, 3.0f)).item<bool>());
    CHECK(array_equal(sum(x, 1), array({3.0f, 6.0f}, {2})).item<bool>());
  }

  // Test prod
  {
    auto x = array({});
    CHECK_EQ(prod(x).item<float>(), 1.0f);

    x = array({2, 2, 2});
    CHECK_EQ(prod(x).item<int>(), 8);
    CHECK(array_equal(prod(x, std::vector<int>{}), x).item<bool>());

    x = full({2, 3}, 2.0f);
    CHECK(array_equal(prod(x, 1), full({2}, 8.0f)).item<bool>());
    CHECK(array_equal(prod(x, 0), full({3}, 4.0f)).item<bool>());
    CHECK_EQ(prod(x, {0, 1}).item<float>(), 64.0f);

    x = full({2, 3, 4}, 2.0f);
    CHECK(array_equal(prod(x, 0), full({3, 4}, 4.0f)).item<bool>());
    CHECK(array_equal(prod(x, 1), full({2, 4}, 8.0f)).item<bool>());
    CHECK(array_equal(prod(x, 2), full({2, 3}, 16.0f)).item<bool>());
    CHECK(array_equal(prod(x, {0, 1}), full({4}, 64.0f)).item<bool>());
    CHECK(array_equal(prod(x, {0, 2}), full({3}, 256.0f)).item<bool>());
    CHECK(array_equal(prod(x, {1, 2}), full({2}, 4096.0f)).item<bool>());

    // Tests with non-uniform results after reduction
    x = array({1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f}, {2, 3});
    CHECK(array_equal(prod(x, 0), full({3}, 2.0f)).item<bool>());
    CHECK(array_equal(prod(x, 1), array({1.0f, 8.0f}, {2})).item<bool>());

    x = array({true, true, true, false, true, false}, {2, 3});
    CHECK(array_equal(prod(x, 0), array({false, true, false})).item<bool>());
    CHECK(array_equal(prod(x, 1), array({true, false})).item<bool>());
  }

  // Test all
  {
    auto x = array({});
    CHECK_EQ(all(x).item<bool>(), true);

    x = array({2, 2, 2});
    CHECK_EQ(all(x).item<bool>(), true);
    auto out = all(x, std::vector<int>{});
    CHECK(array_equal(out, array({true, true, true})).item<bool>());

    x = array({0, 2, 2});
    CHECK_EQ(all(x).item<bool>(), false);

    x = array({true, true, true, false, true, false}, {2, 3});
    CHECK(array_equal(all(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(all(x, 0), array({false, true, false})).item<bool>());
  }

  // Test any
  {
    auto x = array({});
    CHECK_EQ(any(x).item<bool>(), false);

    x = array({0, 0, 0});
    CHECK_EQ(any(x).item<bool>(), false);

    x = array({0, 2, 0});
    CHECK_EQ(any(x).item<bool>(), true);
    auto out = any(x, std::vector<int>{});
    CHECK(array_equal(out, array({false, true, false})).item<bool>());

    x = array({true, false, true, false, false, false}, {2, 3});
    CHECK(array_equal(any(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(any(x, 0), array({true, false, true})).item<bool>());
  }

  // Test max and min
  {
    auto x = array({});
    CHECK_THROWS(max(x));
    CHECK_THROWS(min(x));

    x = array({1.0f, 2.0f, 3.0f});
    CHECK_EQ(max(x).item<float>(), 3.0f);
    CHECK_EQ(min(x).item<float>(), 1.0f);

    x = array({-2.0f, -1.0f});
    CHECK_EQ(max(x).item<float>(), -1.0f);
    CHECK_EQ(min(x).item<float>(), -2.0f);

    constexpr float inf = std::numeric_limits<float>::infinity();
    x = array({inf});
    CHECK_EQ(min(x).item<float>(), inf);

    x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
    CHECK(array_equal(max(x, 0), array({4.0f, 5.0f, 6.0f})).item<bool>());
    CHECK(array_equal(max(x, 1), array({3.0f, 6.0f})).item<bool>());
    CHECK(array_equal(min(x, 0), array({1.0f, 2.0f, 3.0f})).item<bool>());
    CHECK(array_equal(min(x, 1), array({1.0f, 4.0f})).item<bool>());

    x = array({1u, 2u, 3u});
    CHECK_EQ(max(x).item<uint32_t>(), 3u);
    CHECK_EQ(min(x).item<uint32_t>(), 1u);

    x = array({1u, 2u, 3u, 4u, 5u, 6u}, {2, 3});
    CHECK(array_equal(max(x, 0), array({4u, 5u, 6u})).item<bool>());
    CHECK(array_equal(max(x, 1), array({3u, 6u})).item<bool>());
    CHECK(array_equal(min(x, 0), array({1u, 2u, 3u})).item<bool>());
    CHECK(array_equal(min(x, 1), array({1u, 4u})).item<bool>());

    x = array({true, false, true, false, false, false}, {2, 3});
    CHECK(array_equal(max(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(max(x, 0), array({true, false, true})).item<bool>());

    x = array({true, true, true, false, true, false}, {2, 3});
    CHECK(array_equal(min(x, 1), array({true, false})).item<bool>());
    CHECK(array_equal(min(x, 0), array({false, true, false})).item<bool>());
  }

  // Test logsumexp
  {
    auto x = array({});
    CHECK_THROWS(logsumexp(x));

    constexpr float inf = std::numeric_limits<float>::infinity();

    x = array({-inf, -inf});
    CHECK_EQ(logsumexp(x).item<float>(), -inf);

    x = array({0.0f, -inf});
    CHECK_EQ(logsumexp(x).item<float>(), 0.0f);

    x = array({0.0f, inf});
    CHECK_EQ(logsumexp(x).item<float>(), inf);

    x = reshape(arange(6, float32), {2, 3});

    std::vector<float> nums = {0.0f, 1.0f, 2.0f, 3.0f};
    x = array(nums.data(), {2, 2});
    auto y = logsumexp(x, {0, 1}, true);
    CHECK_EQ(y.shape(), Shape{1, 1});
    auto result = std::log(
        std::exp(nums[0]) + std::exp(nums[1]) + std::exp(nums[2]) +
        std::exp(nums[3]));
    CHECK(y.item<float>() == doctest::Approx(result));
    auto expected = array(
        {std::log(std::exp(nums[0]) + std::exp(nums[2])),
         std::log(std::exp(nums[1]) + std::exp(nums[3]))});
    CHECK(allclose(logsumexp(x, 0), expected).item<bool>());

    expected = array(
        {std::log(std::exp(nums[0]) + std::exp(nums[1])),
         std::log(std::exp(nums[2]) + std::exp(nums[3]))});
    CHECK(allclose(logsumexp(x, 1), expected).item<bool>());
  }

  // Test softmax
  {
    for (auto t : {float16, bfloat16, float32}) {
      const auto rtol = t == float32 ? 1e-5 : 1e-2;
      auto x = array({}, t);
      CHECK(array_equal(x, softmax(x)).item<bool>());

      // all zeros
      x = array({0., 0., 0., 0.}, t);
      auto y = array({0.25, 0.25, 0.25, 0.25}, t);
      CHECK(array_equal(y, softmax(x)).item<bool>());
      CHECK(array_equal(y, softmax(x, -1)).item<bool>());
      CHECK(array_equal(y, softmax(x, std::vector<int>{-1})).item<bool>());
      CHECK(array_equal(y, softmax(x, std::vector<int>{0})).item<bool>());

      auto ones = array(1.0f, t);
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());

      // all ones
      x = array({1., 1., 1., 1.}, t);
      CHECK(array_equal(y, softmax(x)).item<bool>());
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());

      // negative values
      x = array({-1., -2., -3., -4.}, t);
      y = array({0.643914, 0.236883, 0.0871443, 0.0320586}, t);
      CHECK(allclose(y, softmax(x), rtol).item<bool>());
      CHECK(allclose(ones, sum(softmax(x)), rtol).item<bool>());

      // positive and negative values
      x = array({1., 0., -1., 0.}, t);
      y = array({0.534447, 0.196612, 0.0723295, 0.196612}, t);
      CHECK(allclose(y, softmax(x), rtol).item<bool>());
      CHECK(allclose(ones, sum(softmax(x)), rtol).item<bool>());

      // large positive values
      x = array({1000., 1000., 1000.}, t);
      y = array({0.333333, 0.333333, 0.333333}, t);
      CHECK(allclose(y, softmax(x)).item<bool>());
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());

      // large negative values
      x = negative(x);
      CHECK(allclose(y, softmax(x)).item<bool>());
      CHECK(array_equal(ones, sum(softmax(x))).item<bool>());
    }
  }
}

TEST_CASE("test irregular binary ops") {
  // 1D strided
  {
    auto x = full({128}, 1.0f);
    auto y = full({64}, 1.0f);
    x = slice(x, {0}, {128}, {4});
    y = slice(y, {0}, {64}, {2});
    CHECK(array_equal(add(x, y), full({32}, 2.0f)).item<bool>());
  }

  // 2D broadcasts
  {
    auto x = full({32, 32}, 4.0f);
    auto y = full({32}, 4.0f);
    CHECK(array_equal(add(x, y), full({32, 32}, 8.0f)).item<bool>());
    y = reshape(y, {32, 1});
    CHECK(array_equal(add(x, y), full({32, 32}, 8.0f)).item<bool>());
    CHECK(array_equal(subtract(y, x), zeros({32, 32})).item<bool>());
  }
}

TEST_CASE("test arithmetic unary ops") {
  // Test negative
  {
    array x(1.0f);
    CHECK_EQ(negative(x).item<float>(), -1.0f);
    CHECK_EQ((-x).item<float>(), -1.0f);

    // works on empty array
    CHECK(array_equal(-array({}), array({})).item<bool>());

    // Throws on bool
    CHECK_THROWS(negative(array(true)));
  }

  // Test logical not
  {
    array x(false);
    CHECK_EQ(logical_not(x).item<bool>(), true);

    x = array(1.0f);
    auto y = logical_not(x);
    CHECK_EQ(y.dtype(), bool_);
    CHECK_EQ(y.item<bool>(), false);

    x = array(0);
    y = logical_not(x);
    CHECK_EQ(y.dtype(), bool_);
    CHECK_EQ(y.item<bool>(), true);
  }

  // Test logical and
  {
    array x(true);
    array y(true);
    CHECK_EQ(logical_and(x, y).item<bool>(), true);

    x = array(1.0f);
    y = array(1.0f);
    auto z = logical_and(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), true);

    x = array(0);
    y = array(1.0f);
    z = logical_and(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), false);
  }

  // Test logical or
  {
    array x(false);
    array y(false);
    CHECK_EQ(logical_or(x, y).item<bool>(), false);

    x = array(1.0f);
    y = array(1.0f);
    auto z = logical_or(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), true);

    x = array(0);
    y = array(1.0f);
    z = logical_or(x, y);
    CHECK_EQ(z.dtype(), bool_);
    CHECK_EQ(z.item<bool>(), true);
  }

  // Test abs
  {
    array x({-1.0f, 0.0f, 1.0f});
    CHECK(array_equal(abs(x), array({1.0f, 0.0f, 1.0f})).item<bool>());

    // works on empty array
    CHECK(array_equal(abs(array({})), array({})).item<bool>());

    // int32
    x = array({-1, 0, 1});
    CHECK(array_equal(abs(x), array({1, 0, 1})).item<bool>());

    // uint32
    x = array({1u, 0u, 1u});
    CHECK(array_equal(abs(x), array({1u, 0u, 1u})).item<bool>());

    // bool
    x = array({false, true});
    CHECK(array_equal(abs(x), array({false, true})).item<bool>());
  }

  // Test sign
  {
    array x({-1.0f, 0.0f, 1.0f});
    CHECK(array_equal(sign(x), x).item<bool>());

    // works on empty array
    CHECK(array_equal(sign(array({})), array({})).item<bool>());

    // int32
    x = array({-1, 0, 1});
    CHECK(array_equal(sign(x), x).item<bool>());

    // uint32
    x = array({1u, 0u, 1u});
    CHECK(array_equal(sign(x), x).item<bool>());

    // bool
    x = array({false, true});
    CHECK(array_equal(sign(x), x).item<bool>());
  }

  constexpr float neginf = -std::numeric_limits<float>::infinity();

  // Test floor and ceil
  {
    array x(1.0f);
    CHECK_EQ(floor(x).item<float>(), 1.0f);
    CHECK_EQ(ceil(x).item<float>(), 1.0f);

    x = array(1.5f);
    CHECK_EQ(floor(x).item<float>(), 1.0f);
    CHECK_EQ(ceil(x).item<float>(), 2.0f);

    x = array(-1.5f);
    CHECK_EQ(floor(x).item<float>(), -2.0f);
    CHECK_EQ(ceil(x).item<float>(), -1.0f);

    x = array(neginf);
    CHECK_EQ(floor(x).item<float>(), neginf);
    CHECK_EQ(ceil(x).item<float>(), neginf);

    x = array(std::complex<float>(1.0f, 1.0f));
    CHECK_THROWS_AS(floor(x), std::invalid_argument);
    CHECK_THROWS_AS(ceil(x), std::invalid_argument);
  }

  // Test round
  {
    array x({0.5, -0.5, 1.5, -1.5, 2.3, 2.6});
    CHECK(array_equal(round(x), array({0, -0, 2, -2, 2, 3})).item<bool>());

    x = array({11, 222, 32});
    CHECK(array_equal(round(x, -1), array({10, 220, 30})).item<bool>());
  }

  // Test exponential
  {
    array x(0.0);
    CHECK_EQ(exp(x).item<float>(), 1.0);

    x = array(2.0);
    CHECK_EQ(exp(x).item<float>(), doctest::Approx(std::exp(2.0f)));

    CHECK(array_equal(exp(array({})), array({})).item<bool>());

    x = array(neginf);
    CHECK_EQ(exp(x).item<float>(), doctest::Approx(0.0f));

    // Integer input type
    x = array(2);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(exp(x).item<float>(), doctest::Approx(std::exp(2.0f)));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(allclose(exp(x), full({2, 2, 2}, std::exp(1.0f))).item<bool>());

    x = split(array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::exp(0.0f), std::exp(2.0f)}, {2, 1});
    CHECK(allclose(exp(x), expected).item<bool>());
  }

  // Test expm1
  {
    array x(-1.0f);
    CHECK_EQ(expm1(x).item<float>(), doctest::Approx(std::expm1(-1.0f)));

    x = array(1.0f);
    CHECK_EQ(expm1(x).item<float>(), doctest::Approx(std::expm1(1.0f)));

    // Integer input type
    x = array(1);
    CHECK_EQ(expm1(x).dtype(), float32);
    CHECK_EQ(expm1(x).item<float>(), doctest::Approx(std::expm1(1.0f)));
  }

  // Test sine
  {
    array x(0.0);
    CHECK_EQ(sin(x).item<float>(), 0.0);

    x = array(M_PI_2);
    CHECK(sin(x).item<float>() == doctest::Approx(std::sin(M_PI_2)));

    CHECK(array_equal(sin(array({})), array({})).item<bool>());

    // Integer input type
    x = array(0);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(sin(x).item<float>(), std::sin(0.0f));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(allclose(sin(x), full({2, 2, 2}, std::sin(1.0f))).item<bool>());

    x = split(array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::sin(0.0f), std::sin(2.0f)}, {2, 1});
    CHECK(allclose(sin(x), expected).item<bool>());
  }

  // Test cos
  {
    array x(0.0);
    CHECK_EQ(cos(x).item<float>(), doctest::Approx(1.0));

    x = array(M_PI_2);
    CHECK(cos(x).item<float>() == doctest::Approx(std::cos(M_PI_2)));

    CHECK(array_equal(cos(array({})), array({})).item<bool>());

    // Integer input type
    x = array(0);
    CHECK_EQ(x.dtype(), int32);
    CHECK(cos(x).item<float>() == doctest::Approx(std::cos(0.0f)));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(allclose(cos(x), full({2, 2, 2}, std::cos(1.0f))).item<bool>());

    x = split(array({0.0f, 1.0f, 2.0f, 3.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::cos(0.0f), std::cos(2.0f)}, {2, 1});
    CHECK(allclose(cos(x), expected).item<bool>());
  }

  // Test degrees
  {
    array x(0.0);
    CHECK_EQ(degrees(x).item<float>(), 0.0);

    x = array(M_PI_2);
    CHECK(degrees(x).item<float>() == doctest::Approx(90.0));

    CHECK(array_equal(degrees(array({})), array({})).item<bool>());

    // Integer input type
    x = array(0);
    CHECK_EQ(x.dtype(), int32);
    CHECK_EQ(degrees(x).item<float>(), 0.0);

    // Input is irregularly strided
    x = broadcast_to(array(M_PI_2), {2, 2, 2});
    CHECK(allclose(degrees(x), full({2, 2, 2}, 90.0)).item<bool>());

    float angles[] = {0.0f, M_PI_2, M_PI, 3.0f * M_PI_2};
    x = split(array(angles, {2, 2}), 2, 1)[0];
    auto expected = array({0.0f, 180.0f}, {2, 1});
    CHECK(allclose(degrees(x), expected).item<bool>());
  }

  // Test radians
  {
    array x(0.0);
    CHECK_EQ(radians(x).item<float>(), 0.0);

    x = array(90.0);
    CHECK(radians(x).item<float>() == doctest::Approx(M_PI_2));

    CHECK(array_equal(radians(array({})), array({})).item<bool>());

    // Integer input type
    x = array(90);
    CHECK_EQ(x.dtype(), int32);
    CHECK(radians(x).item<float>() == doctest::Approx(M_PI_2));

    // Input is irregularly strided
    x = broadcast_to(array(90.0f), {2, 2, 2});
    CHECK(allclose(radians(x), full({2, 2, 2}, M_PI_2)).item<bool>());

    x = split(array({0.0f, 90.0f, 180.0f, 270.0f}, {2, 2}), 2, 1)[0];
    float angles[] = {0.0f, M_PI};
    auto expected = array(angles, {2, 1});
    CHECK(allclose(radians(x), expected).item<bool>());
  }

  // Test log
  {
    array x(0.0);
    CHECK_EQ(log(x).item<float>(), neginf);

    x = array(1.0);
    CHECK_EQ(log(x).item<float>(), log(1.0f));

    // Integer input type
    x = array(1);
    CHECK_EQ(log(x).dtype(), float32);
    CHECK_EQ(log(x).item<float>(), log(1.0f));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(array_equal(log(x), full({2, 2, 2}, std::log(1.0f))).item<bool>());

    x = split(array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::log(1.0f), std::log(3.0f)}, {2, 1});
    CHECK(array_equal(log(x), expected).item<bool>());
  }

  // Test log2
  {
    array x(0.0);
    CHECK_EQ(log2(x).item<float>(), neginf);

    x = array(1.0);
    CHECK_EQ(log2(x).item<float>(), 0.0f);

    x = array(1024.0f);
    CHECK_EQ(log2(x).item<float>(), 10.0f);
  }

  // Test log10
  {
    array x(0.0);
    CHECK_EQ(log10(x).item<float>(), neginf);

    x = array(1.0);
    CHECK_EQ(log10(x).item<float>(), 0.0f);

    x = array(1000.0f);
    CHECK_EQ(log10(x).item<float>(), 3.0f);
  }

  // Test log1p
  {
    array x(-1.0f);
    CHECK_EQ(log1p(x).item<float>(), neginf);

    x = array(1.0f);
    CHECK_EQ(log1p(x).item<float>(), std::log1pf(1.0f));

    // Integer input type
    x = array(1);
    CHECK_EQ(log1p(x).dtype(), float32);
    CHECK_EQ(log1p(x).item<float>(), std::log1pf(1.0f));

    // Input is irregularly strided
    x = broadcast_to(array(1.0f), {2, 2, 2});
    CHECK(
        array_equal(log1p(x), full({2, 2, 2}, std::log1pf(1.0f))).item<bool>());

    x = split(array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), 2, 1)[0];
    auto expected = array({std::log1pf(1.0f), std::log1pf(3.0f)}, {2, 1});
    CHECK(array_equal(log1p(x), expected).item<bool>());
  }

  // Test sigmoid
  {
    array x(0.0);
    CHECK_EQ(sigmoid(x).item<float>(), 0.5f);

    // Integer input type
    x = array(0);
    CHECK_EQ(sigmoid(x).dtype(), float32);
    CHECK_EQ(sigmoid(x).item<float>(), 0.5f);

    constexpr auto inf = std::numeric_limits<float>::infinity();
    x = array(inf);
    CHECK_EQ(sigmoid(x).item<float>(), 1.0f);
    x = array(-inf);
    CHECK_EQ(sigmoid(x).item<float>(), 0.0f);
  }

  // Test square
  {
    array x(3.0);
    CHECK_EQ(square(x).item<float>(), 9.0);

    x = array(2);
    CHECK_EQ(square(x).item<int>(), 4);

    x = full({3, 3}, 2.0f);
    CHECK(array_equal(square(x), full({3, 3}, 4.0f)).item<bool>());
  }

  // Test sqrt and rsqrt
  {
    array x(4.0);
    CHECK_EQ(sqrt(x).item<float>(), 2.0);
    CHECK_EQ(rsqrt(x).item<float>(), 0.5);

    x = full({3, 3}, 9.0f);
    CHECK(array_equal(sqrt(x), full({3, 3}, 3.0f)).item<bool>());

    x = array(4, int32);
    CHECK_EQ(sqrt(x).item<float>(), 2.0f);
    CHECK_EQ(rsqrt(x).item<float>(), 0.5f);
  }

  // Test reciprocal
  {
    array x(8.0);
    CHECK_EQ(reciprocal(x).item<float>(), 0.125f);

    x = array(2);
    auto out = reciprocal(x);
    CHECK_EQ(out.dtype(), float32);
    CHECK_EQ(out.item<float>(), 0.5f);

    x = full({3, 3}, 2.0f);
    CHECK(array_equal(reciprocal(x), full({3, 3}, 0.5f)).item<bool>());
  }
}

TEST_CASE("test error functions") {
  constexpr float inf = std::numeric_limits<float>::infinity();
  array x(0.0f);
  CHECK_EQ(erf(x).item<float>(), 0.0f);
  x = array(inf);
  CHECK_EQ(erf(x).item<float>(), 1.0f);
  x = array(-inf);
  CHECK_EQ(erf(x).item<float>(), -1.0f);

  x = array(1, int32);
  CHECK_EQ(erf(x).dtype(), float32);

  x = array(0.0f);
  CHECK_EQ(erfinv(x).item<float>(), 0.0f);
  x = array(1.0f);
  CHECK_EQ(erfinv(x).item<float>(), inf);
  x = array(-1.0f);
  CHECK_EQ(erfinv(x).item<float>(), -inf);

  x = array(1, int32);
  CHECK_EQ(erfinv(x).dtype(), float32);

  x = array(2.0f);
  CHECK(std::isnan(erfinv(x).item<float>()));
  x = array(-2.0f);
  CHECK(std::isnan(erfinv(x).item<float>()));

  auto vals = {0.9f, 0.5f, 0.1f, -0.1f, -0.5f, -0.9f};
  // Expected values are generated from scipy's error function:
  //   python -c "import scipy.special as ss;
  //   vals = [0.9, 0.5, 0.1, -0.1, -0.5, -0.9];
  //   print([ss.erf(x) for x in vals])"
  {
    auto expected = {
        0.7969082124228322,
        0.5204998778130465,
        0.1124629160182849,
        -0.1124629160182849,
        -0.5204998778130465,
        -0.7969082124228322};
    for (int i = 0; i < vals.size(); ++i) {
      x = array(vals.begin()[i]);
      CHECK_EQ(erf(x).item<float>(), doctest::Approx(expected.begin()[i]));
    }
  }

  // Expected values are generated from scipy's inverse error function:
  //   python -c "import scipy.special as ss;
  //   vals = [0.9, 0.5, 0.1, -0.1, -0.5, -0.9];
  //   print([ss.erfinv(x) for x in vals])"
  {
    auto expected = {
        1.1630871536766738,
        0.4769362762044699,
        0.08885599049425778,
        -0.08885599049425769,
        -0.4769362762044699,
        -1.1630871536766743};
    for (int i = 0; i < vals.size(); ++i) {
      x = array(vals.begin()[i]);
      CHECK_EQ(erfinv(x).item<float>(), doctest::Approx(expected.begin()[i]));
    }
  }

  // float16_t
  {
    array x(0.0f, float16);
    auto out = erf(x);
    CHECK_EQ(out.dtype(), float16);
    CHECK_EQ(out.item<float16_t>(), 0.0f);

    out = erfinv(x);
    CHECK_EQ(out.dtype(), float16);
    CHECK_EQ(out.item<float16_t>(), 0.0f);
  }

  // bfloat
  {
    array x(0.0f, bfloat16);
    auto out = erf(x);
    CHECK_EQ(out.dtype(), bfloat16);
    CHECK_EQ(out.item<bfloat16_t>(), 0.0f);

    out = erfinv(x);
    CHECK_EQ(out.dtype(), bfloat16);
    CHECK_EQ(out.item<float16_t>(), 0.0f);
  }
}

TEST_CASE("test arithmetic binary ops") {
  array x(1.0);
  array y(1.0);
  auto z = add(x, y);
  CHECK_EQ(z.item<float>(), 2.0);
  z = x + y;
  CHECK_EQ(z.item<float>(), 2.0);
  z = add(z, x);
  CHECK_EQ(z.item<float>(), 3.0);
  z.eval(); // No-op
  CHECK_EQ(z.item<float>(), 3.0);

  // Chain a few adds:
  auto out = x;
  for (int i = 0; i < 10; ++i) {
    out = add(out, x);
  }
  CHECK_EQ(out.item<float>(), 11.0);

  // Works for different shapes
  x = array({1.0, 2.0, 3.0}, {1, 3});
  y = array({1.0, 2.0, 3.0}, {1, 3});
  z = add(x, y);
  CHECK_EQ(z.shape(), Shape{1, 3});
  auto eq = array_equal(z, array({2.0, 4.0, 6.0}, {1, 3}));
  CHECK(eq.item<bool>());

  // Works with scalars
  x = array({1.0, 2.0, 3.0}, {1, 3});
  y = x + 2.0;
  CHECK_EQ(y.dtype(), float32);
  eq = array_equal(y, array({3.0, 4.0, 5.0}, {1, 3}));
  CHECK(eq.item<bool>());
  y = 2.0 + x;
  CHECK_EQ(y.dtype(), float32);
  eq = array_equal(y, array({3.0, 4.0, 5.0}, {1, 3}));
  CHECK(eq.item<bool>());

  // Check type promotion
  y = 2 + x;
  CHECK_EQ(y.dtype(), float32);

  y = 2.0 + array({1, 2, 3});
  CHECK_EQ(y.dtype(), float32);
  CHECK(array_equal(y, array({3.0, 4.0, 5.0})).item<bool>());

  // Broadcasting works
  x = broadcast_to(array({1.0}), {10});
  y = broadcast_to(array({2.0}), {10});
  z = add(x, y);
  CHECK(array_equal(z, full({10}, 3.0)).item<bool>());

  x = array({1.0, 2.0}, {1, 2});
  y = array({1.0, 2.0}, {2, 1});
  z = add(x, y);
  CHECK_EQ(z.shape(), Shape{2, 2});
  eq = array_equal(z, array({2.0, 3.0, 3.0, 4.0}, {2, 2}));
  CHECK(eq.item<bool>());

  x = ones({3, 2, 1});
  z = x + 2.0;
  CHECK_EQ(z.shape(), Shape{3, 2, 1});
  eq = array_equal(z, array({3.0, 3.0, 3.0, 3.0, 3.0, 3.0}, {3, 2, 1}));
  CHECK(eq.item<bool>());

  // Works for empty arrays
  x = array({});
  y = array({});
  z = x + y;
  z.eval();
  CHECK_EQ(z.size(), 0);
  CHECK_EQ(z.shape(), Shape{0});

  // Check subtraction
  x = array({3, 2, 1});
  y = array({1, 1, 1});
  CHECK(array_equal(x - y, array({2, 1, 0})).item<bool>());

  // Check multiplication
  x = array({1, 2, 3});
  y = array({2, 2, 2});
  CHECK(array_equal(x * y, array({2, 4, 6})).item<bool>());

  // Check division
  x = array(1);
  y = array(1);
  CHECK_EQ(divide(x, y).item<float>(), 1.0f);

  x = array(1);
  y = array(0.5);
  CHECK_EQ(divide(x, y).item<float>(), 2.0f);

  x = array(1);
  y = array(4);
  CHECK_EQ(divide(x, y).item<float>(), 0.25f);

  x = array(true);
  y = array(true);
  CHECK_EQ(divide(x, y).item<float>(), 1.0f);

  x = array(false);
  y = array(true);
  CHECK_EQ(divide(x, y).item<float>(), 0.0f);

  x = array(true);
  y = array(false);
  CHECK(std::isinf(divide(x, y).item<float>()));

  x = array(false);
  y = array(false);
  CHECK(std::isnan(divide(x, y).item<float>()));

  // Check maximum and minimum
  x = array(1.0f);
  y = array(0.0f);
  CHECK_EQ(maximum(x, y).item<float>(), 1.0f);
  CHECK_EQ(minimum(x, y).item<float>(), 0.0f);
  y = array(2.0f);
  CHECK_EQ(maximum(x, y).item<float>(), 2.0f);
  CHECK_EQ(minimum(x, y).item<float>(), 1.0f);

  // Check logaddexp
  x = array(0.0f);
  y = array(0.0f);
  CHECK_EQ(logaddexp(x, y).item<float>(), std::log(2.0f));

  x = array(0u);
  y = array(10000u);
  CHECK_EQ(logaddexp(x, y).item<float>(), 10000.0f);

  constexpr float inf = std::numeric_limits<float>::infinity();
  x = array(inf);
  y = array(3.0f);
  CHECK_EQ(logaddexp(x, y).item<float>(), inf);

  x = array(-inf);
  y = array(3.0f);
  CHECK_EQ(logaddexp(x, y).item<float>(), 3.0f);

  x = array(-inf);
  y = array(-inf);
  CHECK_EQ(logaddexp(x, y).item<float>(), -inf);

  x = array(inf);
  y = array(inf);
  CHECK_EQ(logaddexp(x, y).item<float>(), inf);

  x = array(-inf);
  y = array(inf);
  CHECK_EQ(logaddexp(x, y).item<float>(), inf);
}

TEST_CASE("test broadcast") {
  auto s = broadcast_shapes({1}, {1, 2});
  CHECK_EQ(s, Shape{1, 2});

  s = broadcast_shapes({1, 2}, {1});
  CHECK_EQ(s, Shape{1, 2});

  s = broadcast_shapes({2, 2}, {});
  CHECK_EQ(s, Shape{2, 2});

  s = broadcast_shapes({}, {1, 1});
  CHECK_EQ(s, Shape{1, 1});

  s = broadcast_shapes({1, 2, 1}, {2});
  CHECK_EQ(s, Shape{1, 2, 2});

  s = broadcast_shapes({2}, {1, 2, 1});
  CHECK_EQ(s, Shape{1, 2, 2});

  s = broadcast_shapes({2, 2, 2}, {1, 2, 1});
  CHECK_EQ(s, Shape{2, 2, 2});

  s = broadcast_shapes({2, 2, 2, 1}, {1, 2, 1});
  CHECK_EQ(s, Shape{2, 2, 2, 1});

  s = broadcast_shapes({0}, {0, 0});
  CHECK_EQ(s, Shape{0, 0});

  CHECK_EQ(broadcast_shapes({}, {0}), Shape{0});

  s = broadcast_shapes({5, 0}, {0, 5, 0});
  CHECK_EQ(s, Shape{0, 5, 0});

  CHECK_EQ(broadcast_shapes({}, {0}), Shape{0});
  CHECK_EQ(broadcast_shapes({1}, {0}), Shape{0});
  CHECK_EQ(broadcast_shapes({1}, {0}), Shape{0});
  CHECK_EQ(broadcast_shapes({1}, {0, 0}), Shape{0, 0});
  CHECK_EQ(broadcast_shapes({1, 1}, {0}), Shape{1, 0});
  CHECK_EQ(broadcast_shapes({1, 1}, {0, 0}), Shape{0, 0});
  CHECK_EQ(broadcast_shapes({2, 1}, {1, 0}), Shape{2, 0});
  CHECK_EQ(broadcast_shapes({2, 1}, {2, 0}), Shape{2, 0});
  CHECK_EQ(broadcast_shapes({2, 1}, {1, 2, 0}), Shape{1, 2, 0});
  CHECK_THROWS_AS(broadcast_shapes({2}, {0}), std::invalid_argument);
  CHECK_THROWS_AS(broadcast_shapes({2, 1}, {0, 0}), std::invalid_argument);

  CHECK_THROWS_AS(broadcast_shapes({3}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(broadcast_shapes({1, 3}, {2}), std::invalid_argument);
  CHECK_THROWS_AS(broadcast_shapes({3}, {1, 2}), std::invalid_argument);
  CHECK_THROWS_AS(
      broadcast_shapes({1, 3, 2}, {1, 2, 2}), std::invalid_argument);

  auto x = full({1, 1}, 2.3f);
  CHECK_EQ(broadcast_to(x, {1, 1}).item<float>(), 2.3f);

  x = broadcast_to(x, {5, 1});
  CHECK_EQ(x.shape(), Shape{5, 1});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 0});

  CHECK_THROWS_AS(broadcast_to(x, {1, 5}), std::invalid_argument);
  x = broadcast_to(x, {5, 5});
  CHECK_EQ(x.shape(), Shape{5, 5});

  x = zeros({2, 1, 2});
  x = broadcast_to(x, {4, 2, 1, 2});
  CHECK_EQ(x.shape(), Shape{4, 2, 1, 2});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 2, 0, 1});

  // Broadcast on empty arrays works as expected
  x = array({});
  CHECK_THROWS_AS(broadcast_to(x, {1}), std::invalid_argument);

  // Broadcast to empty array works as expected
  x = array({1});
  auto y = broadcast_to(x, {0});
  eval(y);
  CHECK_EQ(y.size(), 0);
  CHECK_EQ(y.shape(), Shape{0});

  x = array({1, 2}, {2, 1});
  y = broadcast_to(x, {2, 0});
  eval(y);
  CHECK_EQ(y.size(), 0);
  CHECK_EQ(y.shape(), Shape{2, 0});

  // Check repeat application works
  x = zeros({2});
  x = broadcast_to(broadcast_to(x, {2, 2}), {2, 2});
  CHECK_EQ(x.shape(), Shape{2, 2});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 1});
  x = broadcast_to(broadcast_to(x, {2, 2}), {2, 2, 2});
  CHECK_EQ(x.shape(), Shape{2, 2, 2});
  x.eval();
  CHECK_EQ(x.strides(), Strides{0, 0, 1});

  // Broadcast on transposed array works
  x = array({0, 1, 2, 3, 4, 5}, {2, 3});
  x = broadcast_to(transpose(x), {2, 3, 2});
  CHECK_EQ(x.shape(), Shape{2, 3, 2});
  y = broadcast_to(array({0, 3, 1, 4, 2, 5}, {3, 2}), {2, 3, 2});
  CHECK(array_equal(x, y).item<bool>());

  // Reshape on broadcasted array works
  x = array(1.0);
  x = broadcast_to(x, {2});
  x = reshape(x, {1, 2});
  CHECK(array_equal(x, ones({1, 2})).item<bool>());
}

TEST_CASE("test gather") {
  // Empty input, non-empty indices/slice
  CHECK_THROWS(gather(array({}), array({1}), 0, {1}));

  // More indices than dimensions
  CHECK_THROWS(gather(array(0), array({1}), 0, {1}));

  // Mismatch dimensions and indices
  CHECK_THROWS(gather(array({0}), {array({0})}, {0, 1}, {1}));
  CHECK_THROWS(gather(array({0}), array({0}), -1, {1}));

  // Repeat dimensions
  CHECK_THROWS(
      gather(array({0}, {1, 1}), {array({0}), array({0})}, {0, 0}, {1, 1}));

  // Slice sizes incorrect
  CHECK_THROWS(gather(array({0}), array({0}), 0, {2}));
  CHECK_THROWS(gather(array({0}), array({0}), 0, {0, 0}));
  CHECK_THROWS(gather(array({0}), array({0}), 0, {-1}));

  // Wrong index type
  CHECK_THROWS(gather(array({0}), array({0.0f}), 0, {0}));
  CHECK_THROWS(
      gather(array({0}, {1, 1}), {array({0}), array({0.0f})}, {0, 1}, {1, 1}));

  // Index arrays must be broadcastable
  CHECK_THROWS(gather(
      array({0}, {1, 1}),
      {array({0, 0, 0}, {3}), array({0, 0}, {2})},
      {0, 1},
      {1, 1}));

  // Basic test of correctness with 1D input
  auto x = arange(20);
  auto y = arange(10);
  auto out = gather(x, y, 0, {1});
  CHECK_EQ(out.shape(), Shape{10, 1});
  CHECK(array_equal(reshape(out, {-1}), y).item<bool>());

  out = gather(x, array({15}, uint32), 0, {1});
  CHECK_EQ(out.shape(), Shape{1, 1});
  CHECK_EQ(out.item<int32_t>(), 15);

  // No index gather works
  out = gather(x, {}, std::vector<int>{}, {10});
  CHECK_EQ(out.shape(), Shape{10});
  CHECK(array_equal(out, arange(10)).item<bool>());

  // Basic test of correctness with 2D input
  x = arange(128);
  x = reshape(x, {4, 32});
  y = array({0, 1}, uint32);
  out = gather(x, y, 0, {1, 32});
  CHECK_EQ(out.shape(), Shape{2, 1, 32});
  CHECK(array_equal(reshape(out, {64}), arange(64)).item<bool>());

  x = reshape(x, {64, 2});
  y = array({0}, uint32);
  out = gather(x, y, 0, {64, 1});
  CHECK_EQ(out.shape(), Shape{1, 64, 1});
  CHECK(array_equal(out, reshape(arange(0, 128, 2), {1, 64, 1})).item<bool>());

  // Basic test of correctness with 3D input
  x = arange(256);
  x = reshape(x, {8, 4, 8});
  y = array({0}, uint32);
  out = gather(x, y, 0, {8, 1, 1});
  CHECK_EQ(out.shape(), Shape{1, 8, 1, 1});
  CHECK(
      array_equal(out, reshape(arange(0, 256, 32), {1, 8, 1, 1})).item<bool>());

  x = broadcast_to(array({1, 2}), {20, 2});
  out = gather(x, array({5}), 0, {1, 1});
  CHECK_EQ(out.item<int>(), 1);
  out = gather(x, {array({5}), array({1})}, {0, 1}, {1, 1});
  CHECK_EQ(out.item<int>(), 2);
}

TEST_CASE("test take") {
  // Empty takes
  auto empty = astype(array({}), int32);
  auto z = take(array({1}), empty);
  CHECK_EQ(z.shape(), Shape{0});
  empty = reshape(empty, {1, 0, 1});
  z = take(array({1}), empty);
  CHECK_EQ(z.shape(), Shape{1, 0, 1});

  CHECK_THROWS(take(array({}), array(1)));

  z = take(array({}), empty);
  CHECK_EQ(z.size(), 0);

  // Take a single row
  auto x = reshape(arange(256), {8, 4, 8});
  z = take(x, array({0}, uint32), 0);
  CHECK_EQ(z.shape(), Shape{1, 4, 8});
  z = reshape(z, {32});
  CHECK(array_equal(z, arange(32)).item<bool>());

  z = take(x, array({1}, uint32), 0);
  z = reshape(z, {32});
  CHECK(array_equal(z, arange(32, 64)).item<bool>());

  // Take multiple rows
  x = arange(256);
  x = reshape(x, {8, 4, 8});
  z = take(x, array({0, 1}, uint32), 0);
  z = reshape(z, {64});
  CHECK(array_equal(z, arange(64)).item<bool>());

  // Take along middle axis
  x = reshape(arange(8), {2, 2, 2});
  z = take(x, array({0}), 1);
  CHECK(array_equal(z, array({0, 1, 4, 5}, {2, 1, 2})).item<bool>());

  // Irregular strides test
  auto a = array({1, 2, 3}, float32);
  auto indices = broadcast_to(array(0), {10});
  auto b = take(a, indices);
  CHECK(array_equal(b, ones({10})).item<bool>());

  // Take with 0 dim index
  z = take(array({0, 1, 2}), array(0));
  CHECK_EQ(z.item<int>(), 0);
  CHECK_EQ(z.ndim(), 0);

  // Check take with float indices crashes
  CHECK_THROWS(take(array({}), array({})));
  CHECK_THROWS(take(a, array({1.0, 2.0, 3.0})));

  // Check axis
  a = array({1, 2, 3, 4}, {2, 2});
  CHECK_THROWS(take(a, array({1}), -3));
  CHECK_THROWS(take(a, array({1}), 2));

  // Check negative indices
  a = array({1, 2, 3, 4}, {2, 2});
  CHECK_EQ(take(a, array({-1})).item<int>(), 4);
  CHECK(array_equal(take(a, array({1, -1})), array({2, 4})).item<bool>());
  CHECK(array_equal(take(a, array(-1), 0), array({3, 4})).item<bool>());

  // Check shapes
  a = zeros({2, 1, 1});
  auto out = take(a, array({1}), 0);
  CHECK(array_equal(out, zeros({1, 1, 1})).item<bool>());
  out = take(a, array({0}), 1);
  CHECK(array_equal(out, zeros({2, 1, 1})).item<bool>());
  out = take(a, array({0}), 1);
  CHECK(array_equal(out, zeros({2, 1, 1})).item<bool>());
  a = zeros({1, 2, 1});
  out = take(a, array({0}), 0);
  CHECK(array_equal(out, zeros({1, 2, 1})).item<bool>());
  out = take(a, array({0}), 1);
  CHECK(array_equal(out, zeros({1, 1, 1})).item<bool>());
  out = take(a, array({0, 1}), 1);
  CHECK(array_equal(out, zeros({1, 2, 1})).item<bool>());

  // Indices have wrong shape
  a = zeros({2, 3, 4});
  CHECK_THROWS(take(a, zeros({1, 3, 4}), 1));
  CHECK_THROWS(take(a, zeros({2, 3, 7}), 1));
  CHECK_THROWS(take(a, zeros({2, 3, 2}), 0));
}

TEST_CASE("test take along axis") {
  // No zero dim arrays
  auto a = array(1);
  CHECK_THROWS(take_along_axis(a, array(0), 0));

  // Index and array size mismatches
  a = arange(5);
  CHECK_THROWS(take_along_axis(a, array({1}), 1));
  CHECK_THROWS(take_along_axis(a, array({1}, {1, 1}), 0));
  CHECK_THROWS(take_along_axis(a, array(1), -1));

  auto out = take_along_axis(a, array({1}), 0);
  CHECK_EQ(out.item<int>(), 1);
  out = take_along_axis(a, array({1}), -1);
  CHECK_EQ(out.item<int>(), 1);

  // Empty arrays
  a = reshape(array({}), {1, 0});
  CHECK_THROWS(take_along_axis(a, array({1}), 0));

  out = take_along_axis(a, reshape(array({1}), {1, 1}), 0);
  eval(out); // Make sure it runs
  CHECK_EQ(out.shape(), Shape{1, 0});

  auto inds = reshape(astype(array({}), int32), {1, 0});
  out = take_along_axis(a, inds, 0);
  eval(out); // Make sure it runs
  CHECK_EQ(out.shape(), Shape{1, 0});

  a = array({1, 2, 3, 4}, {2, 2});
  inds = array({0, 1}, {1, 2});
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({1, 4}, {1, 2})).item<bool>());

  inds = array({0, 1, 0, 1, 0, 0, 1, 0}, {4, 2}, int32);
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({1, 4, 1, 4, 1, 2, 3, 2}, {4, 2})).item<bool>());

  inds = array({0, 1}, {2, 1});
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({1, 4}, {2, 1})).item<bool>());

  // Broadcasting works
  inds = array({0}, {1, 1});
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({1, 2}, {1, 2})).item<bool>());
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({1, 3}, {2, 1})).item<bool>());

  inds = array({0, 1, 1, 0, 0, 1}, {2, 3}, int32);
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({1, 2, 2, 3, 3, 4}, {2, 3})).item<bool>());

  a = reshape(arange(8), {2, 2, 2});
  inds = array({0, 1, 0, 0, 1, 0, 0, 1}, {2, 2, 2});
  out = take_along_axis(a, inds, 0);
  CHECK(array_equal(out, array({0, 5, 2, 3, 4, 1, 2, 7}, {2, 2, 2}))
            .item<bool>());
  out = take_along_axis(a, inds, 1);
  CHECK(array_equal(out, array({0, 3, 0, 1, 6, 5, 4, 7}, {2, 2, 2}))
            .item<bool>());
  out = take_along_axis(a, inds, 2);
  CHECK(array_equal(out, array({0, 1, 2, 2, 5, 4, 6, 7}, {2, 2, 2}))
            .item<bool>());
}

TEST_CASE("test put along axis") {
  // No zero dim arrays
  auto a = array(1);
  auto v = array(1);
  CHECK_THROWS(put_along_axis(a, array(0), v, 0));

  // Index and array size mismatches
  a = arange(5);
  CHECK_THROWS(put_along_axis(a, array({1}), array({0}), 1));
  CHECK_THROWS(put_along_axis(a, array({1}, {1, 1}), array({0}), 0));
  CHECK_THROWS(put_along_axis(a, array(1), array(0), -1));

  auto expected = array({0, 0, 2, 3, 4});
  auto out = put_along_axis(a, array({1}), array({0}), 0);
  CHECK(array_equal(out, expected).item<bool>());

  // Empty arrays
  a = reshape(array({}), {1, 0});
  CHECK_THROWS(put_along_axis(a, array({1}), array({0}), 0));

  auto inds = reshape(astype(array({}), int32), {1, 0});
  out = take_along_axis(a, inds, 0);
  eval(out); // Make sure it runs
  CHECK_EQ(out.shape(), Shape{1, 0});

  a = array({1, 2, 3, 4}, {2, 2});
  inds = array({0, 1}, {1, 2});
  out = put_along_axis(a, inds, array({0}), 0);
  expected = array({0, 2, 3, 0}, {2, 2});
  CHECK(array_equal(out, expected).item<bool>());

  inds = array({0, 0, 1, 1}, {2, 2}, int32);
  auto values = array({2, 3, 4, 5}, {2, 2}, int32);
  out = put_along_axis(a, inds, values, 0);
  CHECK(array_equal(out, array({2, 3, 4, 5}, {2, 2})).item<bool>());

  inds = array({0, 1}, {2, 1});
  out = put_along_axis(a, inds, array({0}), 1);
  expected = array({0, 2, 3, 0}, {2, 2});
  CHECK(array_equal(out, expected).item<bool>());
}

TEST_CASE("test scatter") {
  // More indices than dimensions
  CHECK_THROWS(scatter(array(0), array({1}), array(1), 0));

  // Mismatch dimensions and indices
  CHECK_THROWS(scatter(array({0}), {array({0})}, array({1}, {1, 1}), {0, 1}));
  CHECK_THROWS(scatter(array({0}), array({0}), array({1}, {1, 1}), -1));

  // Repeat dimensions
  CHECK_THROWS(scatter(
      array({0}, {1, 1}), {array({0}), array({0})}, array({1}), {0, 0}));

  // Update sizes incorrect
  CHECK_THROWS(scatter(array({0}), array({0}), array({0, 1}), 0));
  CHECK_THROWS(scatter(array({0}), array({0}), array({0, 1}, {2, 1}), 0));
  CHECK_THROWS(scatter(array({0}, {1}), array({0}), array({0, 1}, {1, 2}), 0));

  // Wrong index type
  CHECK_THROWS(scatter(array({0}), array({0.0f}), array({0}, {1, 1}), 0));
  CHECK_THROWS(scatter(
      array({0}, {1, 1}),
      {array({0}), array({0.0f})},
      array({1}, {1, 1, 1}),
      {0, 1}));

  // Index arrays must be broadcastable
  CHECK_THROWS(scatter(
      array({0}, {1, 1}),
      {array({0, 0, 0}, {3}), array({0, 0}, {2})},
      ones({3, 2, 1, 1}),
      {0, 1}));

  // Single element scatter
  auto in = zeros({4}, float32);
  auto inds = arange(2);
  auto updates = ones({2, 1}, float32);
  auto out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, array({1.0f, 1.0f, 0.0f, 0.0f})).item<bool>());

  // Single element scatter add
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = ones({3, 1}, float32);
  out = scatter_add(in, inds, updates, 0);
  CHECK(array_equal(out, array({3.0f, 1.0f, 1.0f, 2.0f})).item<bool>());

  // Single element scatter prod
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = full({3, 1}, 2.0f, float32);
  out = scatter_prod(in, inds, updates, 0);
  CHECK(array_equal(out, array({4.0f, 1.0f, 1.0f, 2.0f})).item<bool>());

  // Single element scatter max
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = array({1.0f, 6.0f, -2.0f}, {3, 1});
  out = scatter_max(in, inds, updates, 0);
  CHECK(array_equal(out, array({6.0f, 1.0f, 1.0f, 1.0f})).item<bool>());

  // Single element scatter min
  in = ones({4}, float32);
  inds = array({0, 0, 3});
  updates = array({1.0f, -6.0f, 2.0f}, {3, 1});
  out = scatter_min(in, inds, updates, 0);
  CHECK(array_equal(out, array({-6.0f, 1.0f, 1.0f, 1.0f})).item<bool>());

  // Empty scatter
  in = arange(4, float32);
  inds = astype(array({}), uint32);
  updates = reshape(array({}), {0, 1});
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, in).item<bool>());

  // Array scatters
  in = zeros({4, 4}, float32);
  inds = array({0, 1, 2, 3});
  updates = reshape(arange(16, float32), {4, 1, 4});
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, reshape(arange(16, float32), {4, 4})).item<bool>());

  // Array scatters with col contiguous updates
  in = zeros({4, 4}, float32);
  inds = array({0, 1, 2, 3});
  updates = transpose(reshape(arange(16, float32), {4, 1, 4}));
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, transpose(reshape(arange(16, float32), {4, 4})))
            .item<bool>());

  // Irregular strided index and reduce collision test
  in = zeros({10}, float32);
  inds = broadcast_to(array(3), {10});
  updates = ones({10, 1}, float32);
  out = scatter_add(in, inds, updates, 0);
  CHECK_EQ(take(out, array(3)).item<float>(), 10);

  // 1 element array with 0 dim index
  in = array({1}, int32);
  updates = array({2}, int32);
  out = scatter_max(in, array(0), updates, 0);
  CHECK_EQ(out.item<int>(), 2);

  // No index arrays or axes
  out = scatter_max(array(1), {}, array(2), std::vector<int>{});
  CHECK_EQ(out.item<int>(), 2);

  // Irregularly strided updates test
  in = ones({3, 3});
  updates = broadcast_to(array({2, 2, 2}), {1, 3, 3});
  inds = array({0});
  out = scatter(in, inds, updates, 0);
  CHECK(array_equal(out, ones({3, 3}) * 2).item<bool>());

  // Along different axis
  in = zeros({2, 3});
  updates = array({1, 2, 3, 4}, {2, 2, 1});
  inds = array({0, 2});
  out = scatter(in, inds, updates, 1);
  auto expected = array({1, 0, 3, 2, 0, 4}, {2, 3});
  CHECK(array_equal(out, expected).item<bool>());

  // Multiple index arrays
  in = zeros({2, 2});
  updates = array({1, 2}, {2, 1, 1});
  inds = array({0, 1});
  out = scatter(in, {inds, inds}, updates, {0, 1});
  CHECK(array_equal(out, array({1, 0, 0, 2}, {2, 2})).item<bool>());

  // Broadcasted indices
  in = zeros({2, 2});
  updates = array({5, 2, 9, 1}, {2, 2, 1, 1});
  auto inds0 = array({0, 1}, {2, 1});
  auto inds1 = array({0, 1}, {1, 2});
  out = scatter(in, {inds0, inds1}, updates, {0, 1});
  CHECK(array_equal(out, array({5, 2, 9, 1}, {2, 2})).item<bool>());

  // Brodacasted operand
  in = broadcast_to(array({0, 0}), {2, 2});
  updates = array({1, 1}, {2, 1, 1});
  inds = array({0, 1});
  out = scatter_add(in, inds, updates, 0);
  CHECK(array_equal(out, array({1, 0, 1, 0}, {2, 2})).item<bool>());

  // 1D scatter
  {
    auto dst = zeros({2, 4}, int32);
    auto src = reshape(array({1, 2, 3, 4}), {1, 1, 4});
    auto idx = array({1});
    auto expected = reshape(array({0, 0, 0, 0, 1, 2, 3, 4}), {2, 4});
    auto out = scatter(dst, idx, src, 0);
    CHECK(array_equal(out, expected).item<bool>());
  }

  // 1D indices with 2D update
  {
    auto dst = zeros({3, 4}, int32);
    auto indices = {array({1}), array({2})};
    auto axes = {0, 1};
    auto updates = reshape(array({1, 2, 3, 4}, int32), {1, 2, 2});
    auto out = scatter(dst, indices, updates, axes);
    auto expected =
        reshape(array({0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4}), {3, 4});
    CHECK(array_equal(out, expected).item<bool>());
  }
}

TEST_CASE("test is positive infinity") {
  array x(1.0f);
  CHECK_FALSE(isposinf(x).item<bool>());

  array y(std::numeric_limits<float>::infinity());
  CHECK(isposinf(y).item<bool>());

  array z = identity(7);
  CHECK_FALSE(all(isposinf(z)).item<bool>());

  array w = array({1.0f, std::numeric_limits<float>::infinity(), 2.0f});
  CHECK_FALSE(all(isposinf(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isposinf(a).item<bool>());

  array b(std::numeric_limits<float>::infinity(), float16);
  CHECK(isposinf(b).item<bool>());

  array c(std::numeric_limits<float>::infinity(), bfloat16);
  CHECK(isposinf(c).item<bool>());
}

TEST_CASE("test is negative infinity") {
  array x(1.0f);
  CHECK_FALSE(isneginf(x).item<bool>());

  array y(-std::numeric_limits<float>::infinity());
  CHECK(isneginf(y).item<bool>());

  array z = identity(7);
  CHECK_FALSE(all(isneginf(z)).item<bool>());

  array w = array({1.0f, -std::numeric_limits<float>::infinity(), 2.0f});
  CHECK_FALSE(all(isneginf(w)).item<bool>());

  array a(1.0f, bfloat16);
  CHECK_FALSE(isneginf(a).item<bool>());

  array b(-std::numeric_limits<float>::infinity(), float16);
  CHECK(isneginf(b).item<bool>());

  array c(-std::numeric_limits<float>::infinity(), bfloat16);
  CHECK(isneginf(c).item<bool>());
}

TEST_CASE("test scatter types") {
  for (auto t : {bool_, uint8, uint16, int8, int16}) {
    auto in = zeros({4, 4}, t);
    auto inds = {arange(4), arange(4)};
    auto updates = ones({4, 1, 1}, t);
    auto out = scatter(in, inds, updates, {0, 1});
    auto expected =
        array({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, t);
    CHECK(array_equal(out, expected).item<bool>());
  }

  for (auto t : {float16, bfloat16}) {
    auto in = zeros({4, 4}, t);
    auto inds = {arange(4), arange(4)};
    auto updates = ones({4, 1, 1}, t);
    auto out = scatter(in, inds, updates, {0, 1});
    auto expected =
        array({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, {4, 4}, t);
    CHECK(allclose(out, expected).item<bool>());
  }
}

TEST_CASE("test complex ops") {
  //  Creation ops
  {
    auto x = full({2, 2}, complex64_t{1, 1});
    CHECK_EQ(x.dtype(), complex64);
    std::initializer_list<complex64_t> expected = {
        {1, 1}, {1, 1}, {1, 1}, {1, 1}};
    CHECK(array_equal(x, array(expected, {2, 2})).item<bool>());
  }

  // Unary ops
  {
    std::initializer_list<complex64_t> vals = {{0, 1}, {1, 0}, {1, 1}};
    auto x = array(vals);

    auto y = abs(x);
    CHECK_EQ(y.dtype(), float32);
    CHECK(array_equal(y, array({1.0f, 1.0f, std::sqrt(2.0f)})).item<bool>());

    y = negative(x);
    std::initializer_list<complex64_t> expected = {{0, -1}, {-1, 0}, {-1, -1}};
    CHECK(array_equal(y, array(expected)).item<bool>());

    y = exp(x);
    {
      std::initializer_list<complex64_t> expected = {
          {0.54030231, 0.84147098}, {2.71828183, 0.}, {1.46869394, 2.28735529}};
      CHECK(allclose(y, array(expected)).item<bool>());
    }

    y = sin(x);
    {
      std::initializer_list<complex64_t> expected = {
          {0., 1.17520119}, {0.84147098, 0.}, {1.29845758, 0.63496391}};
      CHECK(allclose(y, array(expected)).item<bool>());
    }

    y = cos(x);
    {
      std::initializer_list<complex64_t> expected = {
          {1.54308063, -0.}, {0.54030231, -0.}, {0.83373003, -0.98889771}};
      CHECK(allclose(y, array(expected)).item<bool>());
    }
  }

  // Binary ops
  {
    std::initializer_list<complex64_t> vals_x = {{0, 1}, {1, 0}, {1, 1}};
    auto x = array(vals_x);

    std::initializer_list<complex64_t> vals_y = {{2, 0}, {1, 1}, {0, 1}};
    auto y = array(vals_y);

    auto z = add(x, y);
    {
      std::initializer_list<complex64_t> expected = {{2, 1}, {2, 1}, {1, 2}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }

    z = subtract(x, y);
    {
      std::initializer_list<complex64_t> expected = {{-2, 1}, {0, -1}, {1, 0}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }

    z = multiply(x, y);
    {
      std::initializer_list<complex64_t> expected = {{0, 2}, {1, 1}, {-1, 1}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }

    z = maximum(x, y);
    {
      std::initializer_list<complex64_t> expected = {{2, 0}, {1, 1}, {1, 1}};
      CHECK(array_equal(z, array(expected)).item<bool>());
    }
  }

  // Reductions
  if (default_device() == Device::cpu) {
    std::initializer_list<complex64_t> vals = {{0, 0}, {1, 0}, {0, 1}};
    auto x = array(vals);
    CHECK_EQ(max(x).item<complex64_t>(), complex64_t{1, 0});
    CHECK_EQ(min(x).item<complex64_t>(), complex64_t{0, 0});
    CHECK_EQ(sum(x).item<complex64_t>(), complex64_t{1, 1});
    CHECK_EQ(prod(x).item<complex64_t>(), complex64_t{0, 0});
  }
}

TEST_CASE("test as_strided op") {
  auto x = arange(10);
  auto y = as_strided(x, {3, 3}, {1, 1}, 0);
  auto expected = array({0, 1, 2, 1, 2, 3, 2, 3, 4}, {3, 3});
  CHECK(array_equal(y, expected).item<bool>());

  y = as_strided(x, {3, 3}, {0, 3}, 0);
  expected = array({0, 3, 6, 0, 3, 6, 0, 3, 6}, {3, 3});
  CHECK(array_equal(y, expected).item<bool>());

  x = reshape(x, {2, 5}); // 0 1 2 3 ...
  x = transpose(x, {1, 0}); // 0 5 1 6 2 7 ...
  y = as_strided(x, {3, 3}, {2, 1}, 1);
  expected = array({5, 1, 6, 6, 2, 7, 7, 3, 8}, {3, 3});
  CHECK(array_equal(y, expected).item<bool>());
}

TEST_CASE("test scan op") {
  auto x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});
  auto y = cumsum(x, 1, false, true);
  auto expected = array({1.0f, 3.0f, 6.0f, 4.0f, 9.0f, 15.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, false, false);
  expected = array({0.0f, 1.0f, 3.0f, 0.0f, 4.0f, 9.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, true, true);
  expected = array({6.0f, 5.0f, 3.0f, 15.0f, 11.0f, 6.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, true, false);
  expected = array({5.0f, 3.0f, 0.0f, 11.0f, 6.0f, 0.0f}, {2, 3});
  CHECK(array_equal(y, expected).item<bool>());

  x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});
  y = cumsum(x, 0, false, true);
  expected =
      array({1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, false, true);
  expected =
      array({1.0f, 2.0f, 4.0f, 6.0f, 5.0f, 6.0f, 12.0f, 14.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  x = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});
  y = cumsum(x, 0, true, true);
  expected =
      array({6.0f, 8.0f, 10.0f, 12.0f, 5.0f, 6.0f, 7.0f, 8.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 1, true, true);
  expected =
      array({4.0f, 6.0f, 3.0f, 4.0f, 12.0f, 14.0f, 7.0f, 8.0f}, {2, 2, 2});
  CHECK(array_equal(y, expected).item<bool>());

  x = reshape(x, {4, 2});
  y = cumsum(x, 0, false, false);
  expected = array({0.0f, 0.0f, 1.0f, 2.0f, 4.0f, 6.0f, 9.0f, 12.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = cumsum(x, 0, true, false);
  expected =
      array({15.0f, 18.0f, 12.0f, 14.0f, 7.0f, 8.0f, 0.0f, 0.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());

  // Check the vmap implementation
  auto fun = [](array x) { return cumsum(x, 0, false, true); };
  y = vmap(fun, 0, 0)(x);
  expected = array({1.0f, 3.0f, 3.0f, 7.0f, 5.0f, 11.0f, 7.0f, 15.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());

  y = vmap(fun, 1, 1)(x);
  expected = array({1.0f, 2.0f, 4.0f, 6.0f, 9.0f, 12.0f, 16.0f, 20.0f}, {4, 2});
  CHECK(array_equal(y, expected).item<bool>());
}

TEST_CASE("test pad") {
  auto x = zeros({1, 2, 3});
  CHECK_EQ(pad(x, 1).shape(), Shape{3, 4, 5});
  CHECK_EQ(pad(x, {0, 1}).shape(), Shape{2, 3, 4});
  CHECK_EQ(pad(x, {{1, 1}, {1, 2}, {3, 1}}).shape(), Shape{3, 5, 7});

  x = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  auto padded_x = pad(x, 1);
  auto expected = array(
      {0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       2.0f,
       0.0f,
       0.0f,
       3.0f,
       4.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f},
      {4, 4});
  CHECK(array_equal(padded_x, expected).item<bool>());
}

TEST_CASE("test power") {
  CHECK_EQ(power(array(1), array(2)).item<int>(), 1);
  CHECK_EQ((power(array(-1), array(2))).item<int>(), 1);
  CHECK_EQ((power(array(-1), array(3))).item<int>(), -1);

  CHECK_EQ((power(array(true), array(false))).item<bool>(), true);
  CHECK_EQ((power(array(false), array(false))).item<bool>(), true);
  CHECK_EQ((power(array(true), array(true))).item<bool>(), true);
  CHECK_EQ((power(array(false), array(true))).item<bool>(), false);

  auto x = array(2.0f);
  CHECK_EQ(
      (power(x, array(0.5))).item<float>(),
      doctest::Approx(std::pow(2.0f, 0.5f)));
  CHECK_EQ(power(x, array(2.0f)).item<float>(), 4.0f);

  CHECK(std::isnan((power(array(-1.0f), array(0.5))).item<float>()));

  auto a = complex64_t{0.5, 0.5};
  auto b = complex64_t{0.5, 0.5};
  auto expected = std::pow(a, b);
  auto out = (power(array(a), array(b))).item<complex64_t>();
  CHECK(abs(out.real() - expected.real()) < 1e-7);
  CHECK(abs(out.imag() - expected.imag()) < 1e-7);

  a = complex64_t{-1.2, 0.1};
  b = complex64_t{2.2, 0.0};
  expected = std::pow(a, b);
  out = (power(array(a), array(b))).item<complex64_t>();
  CHECK(abs(out.real() - expected.real()) < 1e-6);
  CHECK(abs(out.imag() - expected.imag()) < 1e-6);
}

TEST_CASE("test where") {
  const float inf = std::numeric_limits<float>::infinity();

  array condition(true);
  array x(1.0f);
  array y(0.0f);
  auto out = where(condition, x, y);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.item<float>(), 1.0f);

  x = array({1, 2}, {2, 1});
  y = array({3, 4}, {1, 2});
  CHECK(array_equal(where(condition, x, y), broadcast_to(x, {2, 2}))
            .item<bool>());

  condition = array(false);
  CHECK(array_equal(where(condition, x, y), broadcast_to(y, {2, 2}))
            .item<bool>());

  condition = array({true, false});
  out = where(condition, x, y);
  auto expected = array({1, 4, 2, 4}, {2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  condition = array({true, false, false, true}, {2, 2});
  out = where(condition, x, y);
  expected = array({1, 4, 3, 2}, {2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  x = array(1);
  y = array(2);
  out = where(condition, x, y);
  expected = array({1, 2, 2, 1}, {2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  condition = array(true);
  x = array({1, 2, 3});
  y = array({3, 6, 13});
  CHECK(array_equal(where(condition, x, y), array({1, 2, 3})).item<bool>());

  condition = array(false);
  x = array({1, 2, 3});
  y = array({3, 6, 13});
  CHECK(array_equal(where(condition, x, y), array({3, 6, 13})).item<bool>());

  condition = array({1, 1, 0});
  x = array({1, 2, 3});
  y = array({11, 12, 13});
  CHECK(array_equal(where(condition, x, y), array({1, 2, 13})).item<bool>());

  condition = array({true, false}, {2, 1, 1});
  x = array({1, 2, 3, 4}, {2, 1, 2});
  y = array({11, 22, 33, 44}, {2, 2, 1});
  expected = array({1, 2, 1, 2, 33, 33, 44, 44}, {2, 2, 2});
  CHECK(array_equal(where(condition, x, y), expected).item<bool>());

  condition = array({true, false, false});
  x = array({inf, 2.0, 3.0});
  y = array({10.0, 20.0, -inf});
  CHECK(array_equal(where(condition, x, y), array({inf, 20.0, -inf}))
            .item<bool>());

  // 4-dim optimized case.
  condition = array({false});
  x = array({1, 2}, {2, 1, 1, 1});
  y = array({3, 4}, {1, 1, 2, 1});
  CHECK(array_equal(where(condition, x, y), array({3, 4, 3, 4}, {2, 1, 2, 1}))
            .item<bool>());

  // 5-dim optimized case.
  condition = array({true, false}, {2, 1, 1, 1, 1});
  x = array({1, 2, 3, 4}, {2, 1, 1, 1, 2});
  y = array({11, 22}, {1, 1, 2, 1, 1});
  CHECK(array_equal(
            where(condition, x, y),
            array({1, 2, 1, 2, 11, 11, 22, 22}, {2, 1, 2, 1, 2}))
            .item<bool>());
}

TEST_CASE("test stack") {
  auto x = array({});
  CHECK_EQ(stack({x}, 0).shape(), Shape{1, 0});
  CHECK_EQ(stack({x}, 1).shape(), Shape{0, 1});

  x = array({1, 2, 3}, {3});
  CHECK_EQ(stack({x}, 0).shape(), Shape{1, 3});
  CHECK_EQ(stack({x}, 1).shape(), Shape{3, 1});

  auto y = array({4, 5, 6}, {3});
  auto z = std::vector<array>{x, y};
  CHECK_EQ(stack(z).shape(), Shape{2, 3});
  CHECK_EQ(stack(z, 0).shape(), Shape{2, 3});
  CHECK_EQ(stack(z, 1).shape(), Shape{3, 2});
  CHECK_EQ(stack(z, -1).shape(), Shape{3, 2});
  CHECK_EQ(stack(z, -2).shape(), Shape{2, 3});

  CHECK_THROWS_MESSAGE(stack({}, 0), "No arrays provided for stacking");

  x = array({1, 2, 3}, {3}, float16);
  y = array({4, 5, 6}, {3}, int32);
  CHECK_EQ(stack({x, y}, 0).dtype(), float16);

  x = array({1, 2, 3}, {3}, int32);
  y = array({4, 5, 6, 7}, {4}, int32);
  CHECK_THROWS_MESSAGE(
      stack({x, y}, 0), "All arrays must have the same shape and dtype");
}

TEST_CASE("test eye") {
  auto eye_3 = eye(3);
  CHECK_EQ(eye_3.shape(), Shape{3, 3});
  auto expected_eye_3 =
      array({1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {3, 3});
  CHECK(array_equal(eye_3, expected_eye_3).item<bool>());

  auto eye_3x2 = eye(3, 2);
  CHECK_EQ(eye_3x2.shape(), Shape{3, 2});
  auto expected_eye_3x2 = array({1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}, {3, 2});
  CHECK(array_equal(eye_3x2, expected_eye_3x2).item<bool>());
}

TEST_CASE("test tri") {
  auto _tri = tri(4, 4, 0, float32);
  CHECK_EQ(_tri.shape(), Shape{4, 4});
  auto expected_tri = array(
      {1.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       1.0f,
       0.0f,
       0.0f,
       1.0f,
       1.0f,
       1.0f,
       0.0f,
       1.0f,
       1.0f,
       1.0f,
       1.0f},
      {4, 4});
  CHECK(array_equal(_tri, expected_tri).item<bool>());
}

TEST_CASE("test tril") {
  auto _tril = tril(full({4, 4}, 2.0f, float32), 0);
  CHECK_EQ(_tril.shape(), Shape{4, 4});
  auto expected_tri = array(
      {2.0f,
       0.0f,
       0.0f,
       0.0f,
       2.0f,
       2.0f,
       0.0f,
       0.0f,
       2.0f,
       2.0f,
       2.0f,
       0.0f,
       2.0f,
       2.0f,
       2.0f,
       2.0f},
      {4, 4});
  CHECK(array_equal(_tril, expected_tri).item<bool>());
}

TEST_CASE("test triu") {
  auto _triu = triu(full({4, 4}, 2.0f, float32), 0);
  CHECK_EQ(_triu.shape(), Shape{4, 4});
  auto expected_tri = array(
      {2.0f,
       2.0f,
       2.0f,
       2.0f,
       0.0f,
       2.0f,
       2.0f,
       2.0f,
       0.0f,
       0.0f,
       2.0f,
       2.0f,
       0.0f,
       0.0f,
       0.0f,
       2.0f},
      {4, 4});
  CHECK(array_equal(_triu, expected_tri).item<bool>());
}

TEST_CASE("test identity") {
  auto id_4 = identity(4);
  CHECK_EQ(id_4.shape(), Shape{4, 4});
  auto expected_id_4 = array(
      {1.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f,
       0.0f,
       0.0f,
       0.0f,
       0.0f,
       1.0f},
      {4, 4});
  CHECK(array_equal(id_4, expected_id_4).item<bool>());
}

TEST_CASE("test eye with positive k offset") {
  auto eye_3_k1 = eye(3, 4, 1);
  CHECK_EQ(eye_3_k1.shape(), Shape{3, 4});
  auto expected_eye_3_k1 = array(
      {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
      {3, 4});
  CHECK(array_equal(eye_3_k1, expected_eye_3_k1).item<bool>());
}

TEST_CASE("test eye with negative k offset") {
  auto eye_4_k_minus1 = eye(4, 3, -1);
  CHECK_EQ(eye_4_k_minus1.shape(), Shape{4, 3});
  auto expected_eye_4_k_minus1 = array(
      {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f},
      {4, 3});
  CHECK(array_equal(eye_4_k_minus1, expected_eye_4_k_minus1).item<bool>());
}

TEST_CASE("test basic clipping") {
  array a({1.0f, 4.0f, 3.0f, 8.0f, 5.0f}, {5});
  array expected({2.0f, 4.0f, 3.0f, 6.0f, 5.0f}, {5});
  auto clipped = clip(a, array(2.0f), array(6.0f));
  CHECK(array_equal(clipped, expected).item<bool>());
}

TEST_CASE("test clipping with only min") {
  array a({-1.0f, 1.0f, 0.0f, 5.0f}, {4});
  array expected({0.0f, 1.0f, 0.0f, 5.0f}, {4});
  auto clipped = clip(a, array(0.0f), std::nullopt);
  CHECK(array_equal(clipped, expected).item<bool>());
}

TEST_CASE("test clipping with only max") {
  array a({2.0f, 3.0f, 4.0f, 5.0f}, {4});
  array expected({2.0f, 3.0f, 4.0f, 4.0f}, {4});
  auto clipped = clip(a, std::nullopt, array(4.0f));
  CHECK(array_equal(clipped, expected).item<bool>());
}

TEST_CASE("test linspace") {
  auto x = linspace(0, 10, 5);
  auto expected = array({0.0f, 2.5f, 5.0f, 7.5f, 10.0f}, {5});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(0, 10, 5, int32);
  expected = array({0, 2, 5, 7, 10}, {5});
  CHECK(array_equal(x, expected).item<bool>());

  x = linspace(0, 1, 0);
  expected = array(std::initializer_list<float>{}, {0});
  CHECK(array_equal(x, expected).item<bool>());
}

TEST_CASE("test quantize dequantize") {
  auto x1 = ones({128, 1});
  auto x2 = expand_dims(arange(0, 512, float32), 0);
  auto x = x1 * x2;

  for (int i = 2; i <= 8; i *= 2) {
    int el_per_int = 32 / i;
    auto [x_q, scales, biases] = quantize(x, 128, i);
    CHECK_EQ(x_q.shape(), Shape{128, 512 / el_per_int});
    CHECK_EQ(scales.shape(), Shape{128, 4});
    CHECK_EQ(biases.shape(), Shape{128, 4});

    auto x_hat = dequantize(x_q, scales, biases, 128, i);
    auto max_diff = max(abs(x - x_hat)).item<float>();
    CHECK(max_diff <= 127.0 / (1 << i));
  }
}

TEST_CASE("test repeat") {
  auto data = array({13, 3, 16, 6, 14, 4, 15, 5, 11, 1, 12, 2}, {3, 2, 2});
  auto repeat_axis_0 = repeat(data, 2, 0);
  auto expected_axis_0 = array(
      {13, 3, 16, 6, 13, 3, 16, 6, 14, 4, 15, 5,
       14, 4, 15, 5, 11, 1, 12, 2, 11, 1, 12, 2},
      {6, 2, 2});

  auto repeat_axis_1 = repeat(data, 2, 1);
  auto expected_axis_1 = array(
      {13, 3, 13, 3, 16, 6, 16, 6, 14, 4, 14, 4,
       15, 5, 15, 5, 11, 1, 11, 1, 12, 2, 12, 2},
      {3, 4, 2});

  auto repeat_axis_2 = repeat(data, 2); // default axis == ndim - 1 == 2
  auto expected_axis_2 = array(
      {13, 13, 3, 3, 16, 16, 6, 6, 14, 14, 4, 4,
       15, 15, 5, 5, 11, 11, 1, 1, 12, 12, 2, 2},
      {24});

  // check output
  CHECK(array_equal(repeat_axis_0, expected_axis_0).item<bool>());
  CHECK(array_equal(repeat_axis_1, expected_axis_1).item<bool>());
  CHECK(array_equal(repeat_axis_2, expected_axis_2).item<bool>());

  auto data_2 = array({1, 3, 2}, {3});
  auto repeat_2 = repeat(data_2, 2, 0);
  auto expected_2 = array({1, 1, 3, 3, 2, 2}, {6});
  CHECK(array_equal(repeat_2, expected_2).item<bool>());

  auto data_3 = array({1, 2, 3, 4, 5, 4, 0, 1, 2}, {3, 3});
  auto repeat_3 = repeat(data_3, 2, 0);
  auto expected_3 =
      array({1, 2, 3, 1, 2, 3, 4, 5, 4, 4, 5, 4, 0, 1, 2, 0, 1, 2}, {6, 3});
  CHECK(array_equal(repeat_3, expected_3).item<bool>());

  // 0 repeats
  auto repeat_4 = repeat(data_3, 0, 0);
  auto expected_4 = array({});
  CHECK(array_equal(repeat_2, expected_2).item<bool>());

  // negative repeats
  CHECK_THROWS_AS(repeat(data_3, -3, 0), std::invalid_argument);
}

TEST_CASE("tile") {
  auto x = array({1, 2, 3}, {3});
  auto y = tile(x, {2});
  auto expected = array({1, 2, 3, 1, 2, 3}, {6});
  CHECK(array_equal(y, expected).item<bool>());
  x = array({1, 2, 3, 4}, {2, 2});
  y = tile(x, {2});
  expected = array({1, 2, 1, 2, 3, 4, 3, 4}, {2, 4});
  CHECK(array_equal(y, expected).item<bool>());
  x = array({1, 2, 3, 4}, {2, 2});
  y = tile(x, {4, 1});
  expected = array({1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, {8, 2});
  CHECK(array_equal(y, expected).item<bool>());

  x = array({1, 2, 3, 4}, {2, 2});
  y = tile(x, {2, 2});
  expected = array({1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4}, {4, 4});
  CHECK(array_equal(y, expected).item<bool>());
  x = array({1, 2, 3}, {3});
  y = tile(x, {2, 2, 2});
  expected = array(
      {1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3},
      {2, 2, 6});
  CHECK(array_equal(y, expected).item<bool>());
}

TEST_CASE("tensordot") {
  auto x = reshape(arange(60.), {3, 4, 5});
  auto y = reshape(arange(24.), {4, 3, 2});
  auto z = tensordot(x, y, {1, 0}, {0, 1});
  auto expected = array(
      {4400, 4730, 4532, 4874, 4664, 5018, 4796, 5162, 4928, 5306}, {5, 2});
  CHECK(array_equal(z, expected).item<bool>());
  x = reshape(arange(360.), {3, 4, 5, 6});
  y = reshape(arange(360.), {6, 4, 5, 3});
  CHECK_THROWS_AS(tensordot(x, y, {2, 1, 3}, {1, 2, 0}), std::invalid_argument);
  x = reshape(arange(60.), {3, 4, 5});
  y = reshape(arange(120.), {4, 5, 6});
  z = tensordot(x, y, 2);
  expected = array(
      {14820.,
       15010.,
       15200.,
       15390.,
       15580.,
       15770.,
       37620.,
       38210.,
       38800.,
       39390.,
       39980.,
       40570.,
       60420.,
       61410.,
       62400.,
       63390.,
       64380.,
       65370.},
      {3, 6});
  CHECK(array_equal(z, expected).item<bool>());
}

TEST_CASE("outer") {
  auto x = arange(1.0, 5.0);
  auto y = arange(1.0, 4.0);
  auto z = outer(x, y);
  auto expected = array(
      {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0, 4.0, 8.0, 12.0}, {4, 3});
  CHECK(array_equal(z, expected).item<bool>());

  x = ones({5});
  y = linspace(-2., 2., 5);
  z = outer(x, y);
  expected = array(
      {-2., -1., 0.,  1.,  2., -2., -1., 0.,  1.,  2., -2., -1., 0.,
       1.,  2.,  -2., -1., 0., 1.,  2.,  -2., -1., 0., 1.,  2.},
      {5, 5});
  CHECK(array_equal(z, expected).item<bool>());
}

TEST_CASE("inner") {
  CHECK_THROWS_AS(
      inner(reshape(arange(5.), {1, 5}), reshape(arange(6.), {2, 3})),
      std::invalid_argument);
  auto x = array({1., 2., 3.});
  auto y = array({0., 1., 0.});
  auto z = inner(x, y);
  CHECK_EQ(z.item<float>(), 2.f);

  x = reshape(arange(24.), {2, 3, 4});
  y = arange(4.);
  z = inner(x, y);
  auto expected = array({14., 38., 62., 86., 110., 134.}, {2, 3});
  CHECK(array_equal(z, expected).item<bool>());

  x = reshape(arange(2.), {1, 1, 2});
  y = reshape(arange(6.), {3, 2});
  z = inner(x, y);
  expected = array({1., 3., 5.}, {1, 1, 3});
  CHECK(array_equal(z, expected).item<bool>());

  z = inner(eye(2), array(7.));
  expected = array({7., 0., 0., 7.}, {2, 2});
  CHECK(array_equal(z, expected).item<bool>());
}

TEST_CASE("test divmod") {
  auto x = array({1, 2, 3});
  auto y = array({1, 1, 1});
  auto out = divmod(x, y);
  CHECK(array_equal(out[0], array({1, 2, 3})).item<bool>());
  CHECK(array_equal(out[1], array({0, 0, 0})).item<bool>());

  x = array({5, 6, 7});
  y = array({2, 2, 2});
  out = divmod(x, y);
  CHECK(array_equal(out[0], array({2, 3, 3})).item<bool>());
  CHECK(array_equal(out[1], array({1, 0, 1})).item<bool>());

  // Siblings should be gone after evaling the graph
  CHECK(out[0].siblings().empty());
  CHECK(out[1].siblings().empty());

  x = array({5.0, 6.0, 7.0});
  y = array({2.0, 2.0, 2.0});
  out = divmod(x, y);
  CHECK(array_equal(out[0], array({2.0, 3.0, 3.0})).item<bool>());
  CHECK(array_equal(out[1], array({1.0, 0.0, 1.0})).item<bool>());

  x = array({1.0}, complex64);
  y = array({2.0}, complex64);
  CHECK_THROWS(divmod(x, y));

  // Check that we can eval on both outputs
  x = array({1.0});
  y = array({2.0});
  out = divmod(x, y);
  eval(out);
  CHECK_EQ(out[0].item<float>(), 0.0);
  CHECK_EQ(out[1].item<float>(), 1.0);

  // Check nested in the graph
  x = array({1.0});
  y = array({2.0});
  out = divmod(x, y);
  auto z = out[0] + out[1];
  CHECK_EQ(z.item<float>(), 1.0);

  // Check that we can still eval when one output goes out of scope
  std::vector<array> out_holder;
  {
    out_holder.push_back(divmod(x, y)[0]);
  }
  eval(out_holder);
  CHECK_EQ(out_holder[0].item<float>(), 0.0);

  // Check that we can still eval when the other output goes out of scope
  out_holder.clear();
  {
    out_holder.push_back(divmod(x, y)[1]);
  }
  eval(out_holder);
  CHECK_EQ(out_holder[0].item<float>(), 1.0);
}

TEST_CASE("test diagonal") {
  auto x = array({0, 1, 2, 3, 4, 5, 6, 7}, {4, 2});
  auto out = diagonal(x);
  CHECK(array_equal(out, array({0, 3}, {2})).item<bool>());

  CHECK_THROWS_AS(diagonal(x, 1, 6, 0), std::out_of_range);
  CHECK_THROWS_AS(diagonal(x, 1, 0, -3), std::out_of_range);

  x = array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {3, 4});
  out = diagonal(x, 2, 1, 0);
  CHECK(array_equal(out, array({8}, {1})).item<bool>());

  out = diagonal(x, -1, 0, 1);
  CHECK(array_equal(out, array({4, 9}, {2})).item<bool>());

  out = diagonal(x, -5, 0, 1);
  eval(out);
  CHECK_EQ(out.shape(), Shape{0});

  x = array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {3, 2, 2});
  out = diagonal(x, 1, 0, 1);
  CHECK(array_equal(out, array({2, 3}, {2, 1})).item<bool>());

  out = diagonal(x, 0, 2, 0);
  CHECK(array_equal(out, array({0, 5, 2, 7}, {2, 2})).item<bool>());

  out = diagonal(x, 1, -1, 0);
  CHECK(array_equal(out, array({4, 9, 6, 11}, {2, 2})).item<bool>());

  x = reshape(arange(16), {2, 2, 2, 2});
  out = diagonal(x, 0, 0, 1);
  CHECK(array_equal(out, array({0, 12, 1, 13, 2, 14, 3, 15}, {2, 2, 2}))
            .item<bool>());

  CHECK_THROWS_AS(diagonal(x, 0, 1, 1), std::invalid_argument);

  x = array({0, 1}, {2});
  CHECK_THROWS_AS(diagonal(x, 0, 0, 1), std::invalid_argument);
}

TEST_CASE("test diag") {
  // To few or too many dimensions
  CHECK_THROWS(diag(array(0.0)));
  CHECK_THROWS(diag(array({0.0}, {1, 1, 1})));

  // Test with 1D array
  auto x = array({0, 1, 2, 3}, {4});
  auto out = diag(x, 0);
  CHECK(
      array_equal(
          out, array({0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3}, {4, 4}))
          .item<bool>());

  out = diag(x, 1);
  CHECK(array_equal(
            out,
            array(
                {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0},
                {5, 5}))
            .item<bool>());

  out = diag(x, -1);
  CHECK(array_equal(
            out,
            array(
                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
                 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0},
                {5, 5}))
            .item<bool>());

  // Test with 2D array
  x = array({0, 1, 2, 3, 4, 5, 6, 7, 8}, {3, 3});
  out = diag(x, 0);
  CHECK(array_equal(out, array({0, 4, 8}, {3})).item<bool>());

  out = diag(x, 1);
  CHECK(array_equal(out, array({1, 5}, {2})).item<bool>());

  out = diag(x, -1);
  CHECK(array_equal(out, array({3, 7}, {2})).item<bool>());
}

TEST_CASE("test issubdtype") {
  const auto cats = {
      complexfloating,
      floating,
      inexact,
      signedinteger,
      unsignedinteger,
      integer,
      number,
      generic};
  const auto types = {
      bool_,
      uint8,
      uint16,
      uint32,
      uint64,
      int8,
      int16,
      int32,
      int64,
      float16,
      float32,
      bfloat16,
      complex64};
  for (const auto& type : types) {
    CHECK(issubdtype(type, type));
    CHECK(issubdtype(type, generic));
    switch (kindof(type)) {
      case Dtype::Kind::b:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK_FALSE(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK_FALSE(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::u:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK_FALSE(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK(issubdtype(type, unsignedinteger));
        CHECK(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::i:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK_FALSE(issubdtype(type, inexact));
        CHECK(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::f:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK(issubdtype(type, floating));
        CHECK(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::c:
        CHECK(issubdtype(type, complexfloating));
        CHECK_FALSE(issubdtype(type, floating));
        CHECK(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
      case Dtype::Kind::V:
        CHECK_FALSE(issubdtype(type, complexfloating));
        CHECK(issubdtype(type, floating));
        CHECK(issubdtype(type, inexact));
        CHECK_FALSE(issubdtype(type, signedinteger));
        CHECK_FALSE(issubdtype(type, unsignedinteger));
        CHECK_FALSE(issubdtype(type, integer));
        CHECK(issubdtype(type, number));
        CHECK(issubdtype(type, generic));
        break;
    }
  }

  for (const auto& type : types) {
    CHECK(issubdtype(type, type));
    CHECK(issubdtype(type, generic));
    for (auto type1 : types) {
      CHECK_EQ(issubdtype(type, type1), type == type1);
    }
  }

  for (const auto& cat : cats) {
    CHECK(issubdtype(cat, cat));
    switch (cat) {
      case Dtype::Category::complexfloating:
        CHECK(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::floating:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK(issubdtype(cat, floating));
        CHECK(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::inexact:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::signedinteger:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::unsignedinteger:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK(issubdtype(cat, unsignedinteger));
        CHECK(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::integer:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::number:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
      case Dtype::Category::generic:
        CHECK_FALSE(issubdtype(cat, complexfloating));
        CHECK_FALSE(issubdtype(cat, floating));
        CHECK_FALSE(issubdtype(cat, inexact));
        CHECK_FALSE(issubdtype(cat, signedinteger));
        CHECK_FALSE(issubdtype(cat, unsignedinteger));
        CHECK_FALSE(issubdtype(cat, integer));
        CHECK_FALSE(issubdtype(cat, number));
        CHECK(issubdtype(cat, generic));
        break;
    }
  }
}

TEST_CASE("test atleast_1d") {
  auto x = array(1);
  auto out = atleast_1d(x);
  CHECK_EQ(out.ndim(), 1);
  CHECK_EQ(out.shape(), Shape{1});

  x = array({1, 2, 3}, {3});
  out = atleast_1d(x);
  CHECK_EQ(out.ndim(), 1);
  CHECK_EQ(out.shape(), Shape{3});

  x = array({1, 2, 3}, {3, 1});
  out = atleast_1d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{3, 1});
}

TEST_CASE("test atleast_1d vector") {
  auto x = std::vector<array>{
      array(1), array({1, 2, 3}, {3}), array({1, 2, 3}, {3, 1})};
  auto out = atleast_1d(x);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].ndim(), 1);
  CHECK_EQ(out[0].shape(), Shape{1});
  CHECK_EQ(out[1].ndim(), 1);
  CHECK_EQ(out[1].shape(), Shape{3});
  CHECK_EQ(out[2].ndim(), 2);
  CHECK_EQ(out[2].shape(), Shape{3, 1});
}

TEST_CASE("test atleast_2d") {
  auto x = array(1);
  auto out = atleast_2d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{1, 1});

  x = array({1, 2, 3}, {3});
  out = atleast_2d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{1, 3});

  x = array({1, 2, 3}, {3, 1});
  out = atleast_2d(x);
  CHECK_EQ(out.ndim(), 2);
  CHECK_EQ(out.shape(), Shape{3, 1});
}

TEST_CASE("test atleast_2d vector") {
  auto x = std::vector<array>{
      array(1), array({1, 2, 3}, {3}), array({1, 2, 3}, {3, 1})};
  auto out = atleast_2d(x);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].ndim(), 2);
  CHECK_EQ(out[0].shape(), Shape{1, 1});
  CHECK_EQ(out[1].ndim(), 2);
  CHECK_EQ(out[1].shape(), Shape{1, 3});
  CHECK_EQ(out[2].ndim(), 2);
  CHECK_EQ(out[2].shape(), Shape{3, 1});
}

TEST_CASE("test atleast_3d") {
  auto x = array(1);
  auto out = atleast_3d(x);
  CHECK_EQ(out.ndim(), 3);
  CHECK_EQ(out.shape(), Shape{1, 1, 1});

  x = array({1, 2, 3}, {3});
  out = atleast_3d(x);
  CHECK_EQ(out.ndim(), 3);
  CHECK_EQ(out.shape(), Shape{1, 3, 1});

  x = array({1, 2, 3}, {3, 1});
  out = atleast_3d(x);
  CHECK_EQ(out.ndim(), 3);
  CHECK_EQ(out.shape(), Shape{3, 1, 1});
}

TEST_CASE("test atleast_3d vector") {
  auto x = std::vector<array>{
      array(1), array({1, 2, 3}, {3}), array({1, 2, 3}, {3, 1})};
  auto out = atleast_3d(x);
  CHECK_EQ(out.size(), 3);
  CHECK_EQ(out[0].ndim(), 3);
  CHECK_EQ(out[0].shape(), Shape{1, 1, 1});
  CHECK_EQ(out[1].ndim(), 3);
  CHECK_EQ(out[1].shape(), Shape{1, 3, 1});
  CHECK_EQ(out[2].ndim(), 3);
  CHECK_EQ(out[2].shape(), Shape{3, 1, 1});
}

TEST_CASE("test topk") {
  auto x = reshape(arange(10), {2, 5});

  {
    auto y = topk(x, 1, 1);
    CHECK(array_equal(y, array({4, 9}, {2, 1})).item<bool>());
  }

  {
    auto y = topk(x, 2, 0);
    CHECK(array_equal(y, x).item<bool>());
  }

  {
    auto y = topk(x, 1, 0);
    CHECK(array_equal(y, array({5, 6, 7, 8, 9}, {1, 5})).item<bool>());
  }
}

TEST_CASE("test meshgrid") {
  // Test default
  auto x = array({1, 2, 3}, {3});
  auto in = std::vector<array>{x};
  auto out = meshgrid(in);
  CHECK(array_equal(out[0], x).item<bool>());

  // Test different lengths
  auto y = array({4, 5}, {2});
  in = std::vector<array>{x, y};
  out = meshgrid(in);
  auto expected_zero = array({1, 2, 3, 1, 2, 3}, {2, 3});
  auto expected_one = array({4, 4, 4, 5, 5, 5}, {2, 3});
  CHECK(array_equal(out[0], expected_zero).item<bool>());
  CHECK(array_equal(out[1], expected_one).item<bool>());

  // Test sparse true
  in = std::vector<array>{x, x};
  out = meshgrid(in, true);
  expected_zero = array({1, 2, 3}, {1, 3});
  expected_one = array({1, 2, 3}, {3, 1});
  CHECK(array_equal(out[0], expected_zero).item<bool>());
  CHECK(array_equal(out[1], expected_one).item<bool>());
}

TEST_CASE("test conv1d") {
  auto in = astype(
      array(
          {0.5488135,
           0.71518937,
           0.60276338,
           0.54488318,
           0.4236548,
           0.64589411},
          {1, 3, 2}),
      float16);

  int kernel = 3;
  int stride = 1;
  int padding = 1;

  {
    int groups = 1;
    auto wt = astype(
        array(
            {

                0.43758721, 0.891773,   0.96366276, 0.38344152,
                0.79172504, 0.52889492,

                0.56804456, 0.92559664, 0.07103606, 0.0871293,
                0.0202184,  0.83261985,

                0.77815675, 0.87001215, 0.97861834, 0.79915856,
                0.46147936, 0.78052918,

                0.11827443, 0.63992102, 0.14335329, 0.94466892,
                0.52184832, 0.41466194

            },
            {4, 3, 2}),
        float16);

    auto expected = array(
        {1.56836,
         0.567383,
         1.8125,
         1.29492,
         2.34375,
         1.61035,
         2.77539,
         1.61328,
         1.40527,
         0.933105,
         1.87402,
         1.09082},
        {1, 3, 4});

    auto out = conv1d(in, wt, stride, padding, /* dilation= */ 1, groups);
    CHECK(allclose(out, expected).item<bool>());
  }

  {
    int groups = 2;
    auto wt = array(
        {0.43758721,
         0.891773,
         0.96366276,

         0.38344152,
         0.79172504,
         0.52889492,

         0.56804456,
         0.92559664,
         0.07103606,

         0.0871293,
         0.0202184,
         0.83261985

        },
        {4, 3, 1});

    auto expected = array(
        {1.07007,
         0.753201,
         0.700818,
         0.468176,
         1.18568,
         0.91152,
         0.956607,
         0.611213,
         0.641404,
         0.566401,
         0.907472,
         0.0605397},
        {1, 3, 4});

    auto out = conv1d(in, wt, stride, padding, /* dilation= */ 1, groups);
    CHECK(allclose(out, expected).item<bool>());
  }
}

TEST_CASE("test conv2d") {
  auto in = array(
      {0.57429284,
       -0.21628855,
       -0.18673691,
       -0.3793517,

       0.3059678,
       -0.8137168,
       0.6168841,
       -0.26912728},
      {1, 2, 2, 2});

  std::pair<int, int> kernel{2, 2};
  std::pair<int, int> stride{1, 1};
  std::pair<int, int> padding{0, 0};

  {
    int groups = 1;

    auto wt = array(
        {0.3190391,   -0.24937038, 1.4621079,   -2.0601406,  -0.3224172,
         -0.38405436, 1.1337694,   -1.0998913,  -0.1724282,  -0.8778584,
         0.04221375,  0.58281523,  -1.1006192,  1.1447237,   0.9015907,
         0.50249434,  0.90085596,  -0.68372786, -0.12289023, -0.93576944,
         -0.26788807, 0.53035545,  -0.69166076, -0.39675352, -0.6871727,
         -0.84520566, -0.6712461,  -0.0126646,  -1.1173104,  0.2344157,
         1.6598022,   0.74204415},
        {4, 2, 2, 2});

    auto expected =
        array({1.9549234, -0.98542136, 0.2097499, 0.20991313}, {1, 1, 1, 4});
    auto out = conv2d(in, wt, stride, padding, /* dilation= */ {1, 1}, groups);
    CHECK(allclose(out, expected).item<bool>());
  }

  {
    int groups = 2;
    auto wt = array(
        {0.3190391,
         -0.24937038,

         1.46210794,
         -2.06014071,

         -0.3224172,
         -0.38405435,

         1.13376944,
         -1.09989127,

         -0.17242821,
         -0.87785842,

         0.04221375,
         0.58281521,

         -1.10061918,
         1.14472371,

         0.90159072,
         0.50249434},
        {4, 2, 2, 1});

    auto expected = array(
        {-0.59372161, -0.44505326, 0.17910982, -1.06507601}, {1, 1, 1, 4});

    auto out = conv2d(in, wt, stride, padding, /* dilation= */ {1, 1}, groups);
    CHECK(allclose(out, expected).item<bool>());
  }

  {
    in = array(
        {0.57429284,
         -0.21628855,
         -0.18673691,
         -0.3793517,

         0.3059678,
         -0.8137168,
         0.6168841,
         -0.26912728,

         0.57429284,
         -0.21628855,
         -0.18673691,
         -0.3793517,

         0.3059678,
         -0.8137168,
         0.6168841,
         -0.26912728},
        {2, 2, 2, 2});

    int groups = 2;
    auto wt = array(
        {0.3190391,
         -0.24937038,

         1.46210794,
         -2.06014071,

         -0.3224172,
         -0.38405435,

         1.13376944,
         -1.09989127,

         -0.17242821,
         -0.87785842,

         0.04221375,
         0.58281521,

         -1.10061918,
         1.14472371,

         0.90159072,
         0.50249434},
        {4, 2, 2, 1});

    auto expected = array(
        {-0.59372161, -0.44505326, 0.17910982, -1.06507601}, {1, 1, 1, 4});

    auto out = conv2d(in, wt, stride, padding, /* dilation= */ {1, 1}, groups);
    CHECK(allclose(out, expected).item<bool>());
  }
}

TEST_CASE("test trace") {
  auto in = eye(3);
  auto out = trace(in).item<float>();
  CHECK_EQ(out, 3.0);

  in = array({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3}, int32);
  auto out2 = trace(in).item<int>();
  CHECK_EQ(out2, 15);

  in = reshape(arange(8), {2, 2, 2});
  auto out3 = trace(in, 0, 0, 1);
  CHECK(array_equal(out3, array({6, 8}, {2})).item<bool>());

  auto out4 = trace(in, 0, 1, 2, float32);
  CHECK(array_equal(out4, array({3, 11}, {2})).item<bool>());
}

TEST_CASE("test view") {
  auto in = array(3);
  CHECK_THROWS(view(in, int64));

  in = array({1, 2, 3});
  CHECK_THROWS(view(in, int64));

  in = array({1, 2, 3, 4}, int64);
  auto out = view(in, int32);
  CHECK(array_equal(out, array({1, 0, 2, 0, 3, 0, 4, 0})).item<bool>());
}

TEST_CASE("test roll") {
  auto x = reshape(arange(10), {2, 5});

  auto y = roll(x, 2);
  CHECK(array_equal(y, array({8, 9, 0, 1, 2, 3, 4, 5, 6, 7}, {2, 5}))
            .item<bool>());

  y = roll(x, -2);
  CHECK(array_equal(y, array({2, 3, 4, 5, 6, 7, 8, 9, 0, 1}, {2, 5}))
            .item<bool>());

  y = roll(x, 2, 1);
  CHECK(array_equal(y, array({3, 4, 0, 1, 2, 8, 9, 5, 6, 7}, {2, 5}))
            .item<bool>());

  y = roll(x, -2, 1);
  CHECK(array_equal(y, array({2, 3, 4, 0, 1, 7, 8, 9, 5, 6}, {2, 5}))
            .item<bool>());

  y = roll(x, 2, {0, 0, 0});
  CHECK(array_equal(y, array({0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {2, 5}))
            .item<bool>());

  y = roll(x, 1, {1, 1, 1});
  CHECK(array_equal(y, array({2, 3, 4, 0, 1, 7, 8, 9, 5, 6}, {2, 5}))
            .item<bool>());

  y = roll(x, {1, 2}, {0, 1});
  CHECK(array_equal(y, array({8, 9, 5, 6, 7, 3, 4, 0, 1, 2}, {2, 5}))
            .item<bool>());
}

TEST_CASE("test contiguous") {
  auto x = array({1, 2, 3});
  x = contiguous(broadcast_to(x, {2, 2, 3}));
  eval(x);
  CHECK(x.flags().row_contiguous);
  CHECK_EQ(x.strides(), decltype(x.strides()){6, 3, 1});

  x = array({1, 2, 1, 2}, {2, 2});
  x = contiguous(transpose(x), true);
  eval(x);
  CHECK(x.flags().col_contiguous);
  CHECK_EQ(x.strides(), decltype(x.strides()){1, 2});
}
