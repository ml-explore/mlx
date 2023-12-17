// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include <cmath>
#include <iostream>
#include "mlx/linalg.h"
#include "mlx/mlx.h"

using namespace mlx::core;
using namespace mlx::core::linalg;

TEST_CASE("vector_norm") {
  //  Test 1-norm on a vector
  CHECK(
      array_equal(vector_norm(ones({3}), 1.0, false), array(3.0)).item<bool>());
  CHECK(array_equal(vector_norm(ones({3}), 1.0, true), array({3.0}))
            .item<bool>());
  //  Test 1-norm on a matrix
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, false), array(36))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, true), array({36}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {0, 1}, false),
            array(36))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {0, 1}, true),
            array({36}, {1, 1}))
            .item<bool>());
  //    Over columns
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {1}, false),
            array({3, 12, 21}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {1}, true),
            array({3, 12, 21}, {3, 1}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {-1}, false),
            array({3, 12, 21}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {-1}, true),
            array({3, 12, 21}, {3, 1}))
            .item<bool>());
  //    Over rows
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {0}, false),
            array({9, 12, 15}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {0}, true),
            array({9, 12, 15}, {1, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {-2}, false),
            array({9, 12, 15}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), 1.0, {-2}, true),
            array({9, 12, 15}, {1, 3}))
            .item<bool>());
  //  Test 1-norm on a 3d tensor
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, false), array(153))
            .item<bool>());
  CHECK(
      array_equal(
          vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, true), array({153}))
          .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {0, 1, 2}, false),
            array(153))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {0, 1, 2}, true),
            array({153}, {1, 1, 1}))
            .item<bool>());
  //    Over last axis
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {2}, false),
            array({3, 12, 21, 30, 39, 48}, {2, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {2}, true),
            array({3, 12, 21, 30, 39, 48}, {2, 3, 1}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {-1}, false),
            array({3, 12, 21, 30, 39, 48}, {2, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {-1}, true),
            array({3, 12, 21, 30, 39, 48}, {2, 3, 1}))
            .item<bool>());
  //    Over middle axis
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {1}, false),
            array({9, 12, 15, 36, 39, 42}, {2, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {1}, true),
            array({9, 12, 15, 36, 39, 42}, {2, 1, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {-2}, false),
            array({9, 12, 15, 36, 39, 42}, {2, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {-2}, true),
            array({9, 12, 15, 36, 39, 42}, {2, 1, 3}))
            .item<bool>());
  //    Over the first axis
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {0}, false),
            array({9, 11, 13, 15, 17, 19, 21, 23, 25}, {3, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {0}, true),
            array({9, 11, 13, 15, 17, 19, 21, 23, 25}, {1, 3, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {-3}, false),
            array({9, 11, 13, 15, 17, 19, 21, 23, 25}, {3, 3}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(18), {2, 3, 3}), 1.0, {-3}, true),
            array({9, 11, 13, 15, 17, 19, 21, 23, 25}, {1, 3, 3}))
            .item<bool>());
  // Test 2-norm on a vector
  CHECK(array_equal(vector_norm({3.0, 4.0}, 2.0, false), array(5.0))
            .item<bool>());
  CHECK(array_equal(vector_norm({3.0, 4.0}, 2.0, true), array({5.0}))
            .item<bool>());
  // Test that 2 is default ord
  CHECK(array_equal(vector_norm({3.0, 4.0}, false), array(5.0)).item<bool>());
  CHECK(array_equal(vector_norm({3.0, 4.0}, true), array({5.0})).item<bool>());
  // Test "inf" norm on a matrix
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "inf", false), array(8.0))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "inf", true), array({8.0}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "inf", {1}, false),
            array({2, 5, 8}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "inf", {1}, true),
            array({2, 5, 8}, {3, 1}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "inf", {0}, false),
            array({6.0, 7.0, 8.0}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "inf", {0}, true),
            array({6, 7, 8}, {1, 3}))
            .item<bool>());
  // Test "-inf" norm on a matrix
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "-inf", false), array(0))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "-inf", true), array({0}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "-inf", {1}, false),
            array({0, 3, 6}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "-inf", {1}, true),
            array({0, 3, 6}, {3, 1}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "-inf", {0}, false),
            array({0, 1, 2}))
            .item<bool>());
  CHECK(array_equal(
            vector_norm(reshape(arange(9), {3, 3}), "-inf", {0}, true),
            array({0, 1, 2}, {1, 3}))
            .item<bool>());
}