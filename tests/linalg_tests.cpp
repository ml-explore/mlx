// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include <cmath>
#include <iostream>
#include "mlx/linalg.h"
#include "mlx/mlx.h"

using namespace mlx::core;
using namespace mlx::core::linalg;

TEST_CASE("[mlx.core.linalg.norm] no ord") {
  array arr_one_d({1, 2, 3});
  array arr_two_d = reshape(arange(9), {3, 3});
  array arr_three_d = reshape(arange(18), {2, 3, 3});

  CHECK(array_equal(norm(arr_one_d), array(sqrt(1 + 4 + 9))).item<bool>());
  CHECK(array_equal(norm(arr_one_d, {0}), array(sqrt(1 + 4 + 9))).item<bool>());
  CHECK(array_equal(
            norm(arr_two_d),
            array(sqrt(
                0 + 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 + 8 * 8)))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, {0}),
            array(
                {sqrt(0 + 3 * 3 + 6 * 6),
                 sqrt(1 + 4 * 4 + 7 * 7),
                 sqrt(2 * 2 + 5 * 5 + 8 * 8)}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, {1}),
            array(
                {sqrt(0 + 1 + 2 * 2),
                 sqrt(3 * 3 + 4 * 4 + 5 * 5),
                 sqrt(6 * 6 + 7 * 7 + 8 * 8)}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, {0, 1}),
            array(sqrt(
                0 + 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 + 8 * 8)))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_three_d, {2}),
            array(
                {
                    sqrt(0 + 1 + 2 * 2),
                    sqrt(3 * 3 + 4 * 4 + 5 * 5),
                    sqrt(6 * 6 + 7 * 7 + 8 * 8),
                    sqrt(9 * 9 + 10 * 10 + 11 * 11),
                    sqrt(12 * 12 + 13 * 13 + 14 * 14),
                    sqrt(15 * 15 + 16 * 16 + 17 * 17),
                },
                {2, 3}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_three_d, {1}),
            array(
                {
                    sqrt(0 + 3 * 3 + 6 * 6),
                    sqrt(1 + 4 * 4 + 7 * 7),
                    sqrt(2 * 2 + 5 * 5 + 8 * 8),
                    sqrt(9 * 9 + 12 * 12 + 15 * 15),
                    sqrt(10 * 10 + 13 * 13 + 16 * 16),
                    sqrt(11 * 11 + 14 * 14 + 17 * 17),
                },
                {2, 3}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_three_d, {0}),
            array(
                {
                    sqrt(0 + 9 * 9),
                    sqrt(1 + 10 * 10),
                    sqrt(2 * 2 + 11 * 11),
                    sqrt(3 * 3 + 12 * 12),
                    sqrt(4 * 4 + 13 * 13),
                    sqrt(5 * 5 + 14 * 14),
                    sqrt(6 * 6 + 15 * 15),
                    sqrt(7 * 7 + 16 * 16),
                    sqrt(8 * 8 + 17 * 17),
                },
                {3, 3}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_three_d, {1, 2}),
            array(
                {sqrt(
                     0 + 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 +
                     8 * 8),
                 sqrt(
                     9 * 9 + 10 * 10 + 11 * 11 + 12 * 12 + 13 * 13 + 14 * 14 +
                     15 * 15 + 16 * 16 + 17 * 17)},
                {2}))
            .item<bool>());
}