// Copyright © 2025 Apple Inc.

#include <numeric>

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test exception from eval_cpu") {
  array a = ones({2, 4}, bfloat16);
  array b = ones({4, 2}, bfloat16);
  array out = matmul(a, b, Device::cpu);
  CHECK_THROWS_WITH_AS(
      eval(out),
      "[Matmul::eval_cpu] Currently only supports float32.",
      std::runtime_error);
}

TEST_CASE("test exception from array.eval") {
  array a = ones({2, 4}, bfloat16);
  array b = ones({4, 2}, bfloat16);
  array out = matmul(a, b, Device::cpu);
  async_eval(out);
  synchronize();
  CHECK_THROWS_WITH_AS(
      out.eval(),
      "[Matmul::eval_cpu] Currently only supports float32.",
      std::runtime_error);
}
