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
