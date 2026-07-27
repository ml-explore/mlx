// Copyright © 2023 Apple Inc.

#include <limits>

#include "doctest/doctest.h"

#include "mlx/backend/common/reduce.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test col reduce grid size") {
  constexpr size_t block_size = 32;
  CHECK_EQ(col_reduce_grid_size(0, block_size), 0);
  CHECK_EQ(col_reduce_grid_size(1, block_size), 1);
  CHECK_EQ(col_reduce_grid_size(31, block_size), 1);
  CHECK_EQ(col_reduce_grid_size(32, block_size), 1);
  CHECK_EQ(col_reduce_grid_size(33, block_size), 2);

  constexpr size_t failing_reduction_stride = size_t{1} << 29;
  constexpr size_t threadgroup_size = 256;
  auto grid_size = col_reduce_grid_size(failing_reduction_stride, block_size);
  CHECK_EQ(grid_size, size_t{1} << 24);
  CHECK_EQ(grid_size * threadgroup_size, size_t{1} << 32);

  CHECK_EQ(
      col_reduce_grid_size(std::numeric_limits<size_t>::max(), block_size),
      std::numeric_limits<size_t>::max() / block_size + 1);
}

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

TEST_CASE("test iinfo") {
  CHECK_EQ(iinfo(int8).dtype, int8);
  CHECK_EQ(iinfo(int64).dtype, int64);
  CHECK_EQ(iinfo(int64).max, std::numeric_limits<int64_t>::max());
  CHECK_EQ(iinfo(uint64).max, std::numeric_limits<uint64_t>::max());
  CHECK_EQ(iinfo(uint64).max, std::numeric_limits<uint64_t>::max());
  CHECK_EQ(iinfo(uint64).min, 0);
  CHECK_EQ(iinfo(int64).min, std::numeric_limits<int64_t>::min());
}
