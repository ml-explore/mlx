// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include <iostream>
#include "mlx/mlx.h"
#include "mlx/linalg.h"
#include "mlx/ops.h"

using namespace mlx::core;

TEST_CASE("test QR factorization") {
  array A = array({{2., 1., 1., 2.}, {2, 2}});
  std::vector<array> out = linalg::qrf(A, default_stream(Device::cpu));
  eval(out);

  array Q = out[0];
  array R = out[1];
  CHECK_EQ(Q.dtype(), float32);
  CHECK_EQ(R.dtype(), float32);
}

// torch
// Q=tensor([[-0.8944, -0.4472],
//         [-0.4472,  0.8944]]),
// R=tensor([[-2.2361, -1.7889],
//         [ 0.0000,  1.3416]]))

// mlx
// 1: Q:array([[-0.894427, -0.447214],
// 1:        [-0.447214, 0.894427]], dtype=float32)
// 1: R:array([[-2.23607, 0.236068],
// 1:        [0, 1.34164]], dtype=float32)
