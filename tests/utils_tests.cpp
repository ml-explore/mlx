// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test type promotion") {
  for (auto t : {bool_, uint32, int32, int64, float32}) {
    auto a = array(0, t);
    CHECK_EQ(result_type({a}), t);

    std::vector<array> arrs = {array(0, t), array(0, t)};
    CHECK_EQ(result_type(arrs), t);
  }

  {
    std::vector<array> arrs = {array(false), array(0, int32)};
    CHECK_EQ(result_type(arrs), int32);
  }

  {
    std::vector<array> arrs = {array(0, int32), array(false), array(0.0f)};
    CHECK_EQ(result_type(arrs), float32);
  }
}

TEST_CASE("test normalize axis") {
  struct TestCase {
    int axis;
    int ndim;
    int expected;
  };

  std::vector<TestCase> testCases = {
      {0, 3, 0}, {1, 3, 1}, {2, 3, 2}, {-1, 3, 2}, {-2, 3, 1}, {-3, 3, 0}};

  for (const auto& tc : testCases) {
    CHECK_EQ(normalize_axis(tc.axis, tc.ndim), tc.expected);
  }

  CHECK_THROWS(normalize_axis(3, 3));
  CHECK_THROWS(normalize_axis(-4, 3));
}

TEST_CASE("test is same size and shape") {
  struct TestCase {
    std::vector<array> a;
    bool expected;
  };

  std::vector<TestCase> testCases = {
      {{array({}), array({})}, true},
      {{array({1}), array({1})}, true},
      {{array({1, 2, 3}), array({1, 2, 4})}, true},
      {{array({1, 2, 3}), array({1, 2})}, false}};

  for (const auto& tc : testCases) {
    CHECK_EQ(is_same_shape(tc.a), tc.expected);
  }
}

TEST_CASE("test check shape dimension") {
  int dim_min = std::numeric_limits<int>::min();
  int dim_max = std::numeric_limits<int>::max();
  CHECK_EQ(check_shape_dim(-4), -4);
  CHECK_EQ(check_shape_dim(0), 0);
  CHECK_EQ(check_shape_dim(12), 12);
  CHECK_EQ(check_shape_dim(static_cast<ssize_t>(dim_min)), dim_min);
  CHECK_EQ(check_shape_dim(static_cast<ssize_t>(dim_max)), dim_max);
  CHECK_EQ(check_shape_dim(static_cast<size_t>(0)), 0);
  CHECK_EQ(check_shape_dim(static_cast<size_t>(dim_max)), dim_max);
  CHECK_THROWS_AS(
      check_shape_dim(static_cast<ssize_t>(dim_min) - 1),
      std::invalid_argument);
  CHECK_THROWS_AS(
      check_shape_dim(static_cast<ssize_t>(dim_max) + 1),
      std::invalid_argument);
  CHECK_THROWS_AS(
      check_shape_dim(static_cast<size_t>(dim_max) + 1), std::invalid_argument);
}
