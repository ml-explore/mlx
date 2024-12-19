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
    CHECK_EQ(normalize_axis_index(tc.axis, tc.ndim), tc.expected);
  }

  CHECK_THROWS(normalize_axis_index(3, 3));
  CHECK_THROWS(normalize_axis_index(-4, 3));
}

TEST_CASE("test finfo") {
  CHECK_EQ(finfo(float32).dtype, float32);
  CHECK_EQ(finfo(complex64).dtype, float32);
  CHECK_EQ(finfo(float16).dtype, float16);
  CHECK_EQ(finfo(float32).min, std::numeric_limits<float>::lowest());
  CHECK_EQ(finfo(float32).max, std::numeric_limits<float>::max());
  CHECK_EQ(finfo(complex64).min, std::numeric_limits<float>::lowest());
  CHECK_EQ(finfo(complex64).max, std::numeric_limits<float>::max());
  CHECK_EQ(finfo(float16).min, -65504);
  CHECK_EQ(finfo(float16).max, 65504);
}
