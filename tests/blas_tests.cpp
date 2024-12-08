// Copyright © 2023 Apple Inc.

#include <numeric>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test matmul") {
  auto a = array(1);
  auto b = array({1.0});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = array({1.0});
  b = array({1.0});
  auto out = matmul(a, b);
  CHECK_EQ(out.shape(), Shape{});
  CHECK_EQ(out.size(), 1);
  CHECK_EQ(out.dtype(), float32);
  CHECK_EQ(out.item<float>(), 1.0f);

  a = ones({2, 4});
  b = ones({2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 3, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4, 3});
  b = ones({4, 2});
  CHECK_THROWS_AS(matmul(a, b), std::invalid_argument);

  a = ones({2, 4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  a = ones({2, 4}, int32);
  b = ones({4, 2}, float32);
  out = matmul(a, b);
  CHECK(array_equal(out, full({2, 2}, 4.0f)).item<bool>());

  // Check single dimensions
  a = ones({4});
  b = ones({4, 2});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({2, 4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({2}, 4.0f)).item<bool>());

  a = ones({4});
  b = ones({4});
  out = matmul(a, b);
  CHECK(array_equal(out, full({}, 4.0f)).item<bool>());

  // Test transposed arrays
  a = array({1.0f, 1.0f, 1.0f, 1.0f}, {1, 4});
  b = array({1.0f, 1.0f, 1.0f, 1.0f}, {4, 1});
  out = matmul(transpose(a), transpose(b));
  CHECK(array_equal(out, ones({4, 4})).item<bool>());

  a = array({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});
  b = array({1.0f, 2.0f, 1.0f, 2.0f}, {2, 2});
  out = matmul(transpose(a), b);
  CHECK(
      array_equal(out, array({4.0f, 8.0f, 6.0f, 12.0f}, {2, 2})).item<bool>());

  out = matmul(a, transpose(b));
  CHECK(
      array_equal(out, array({5.0f, 5.0f, 11.0f, 11.0f}, {2, 2})).item<bool>());

  out = matmul(transpose(a), transpose(b));
  CHECK(
      array_equal(out, array({7.0f, 7.0f, 10.0f, 10.0f}, {2, 2})).item<bool>());

  // Test broadcasting for both arrays
  a = ones({5, 4, 2});
  b = ones({2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 4, 3}, 2.0f)).item<bool>());

  a = ones({5, 1, 4, 2});
  b = ones({1, 7, 2, 3});
  out = matmul(a, b);
  CHECK(array_equal(out, full({5, 7, 4, 3}, 2.0f)).item<bool>());

  // Test batched matmul with transpose
  a = ones({2, 2, 4});
  b = ones({2, 4, 2});
  out = matmul(transpose(a, {0, 2, 1}), transpose(b, {0, 2, 1}));
  CHECK(array_equal(out, full({2, 4, 4}, 2.0f)).item<bool>());
}
