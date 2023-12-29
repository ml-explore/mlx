// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include <iostream>
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test QR factorization") {
  array A = array({{2., 1., 1., 2.}, {2, 2}});
  array out = linalg::qrf(A, default_stream(Device::cpu));
  out.eval();

  // std::cout << out << "\n";
  CHECK_EQ(true, true);
}
