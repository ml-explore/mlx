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
  CHECK(array_equal(norm(arr_one_d, {0}, false), array(sqrt(1 + 4 + 9)))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, {}, false),
            array(sqrt(
                0 + 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 + 8 * 8)))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, {0}, false),
            array(
                {sqrt(0 + 3 * 3 + 6 * 6),
                 sqrt(1 + 4 * 4 + 7 * 7),
                 sqrt(2 * 2 + 5 * 5 + 8 * 8)}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, {1}, false),
            array(
                {sqrt(0 + 1 + 2 * 2),
                 sqrt(3 * 3 + 4 * 4 + 5 * 5),
                 sqrt(6 * 6 + 7 * 7 + 8 * 8)}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, {0, 1}, false),
            array(sqrt(
                0 + 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 + 8 * 8)))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_three_d, {2}, false),
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
            norm(arr_three_d, {1}, false),
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
            norm(arr_three_d, {0}, false),
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
            norm(arr_three_d, {1, 2}, false),
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

TEST_CASE("[mlx.core.linalg.norm] double ord") {
  array arr_one_d({1, 2, 3});
  array arr_two_d = reshape(arange(9), {3, 3});
  array arr_three_d = reshape(arange(18), {2, 3, 3});

  CHECK(array_equal(norm(arr_one_d, 2.0), array(sqrt(1 + 4 + 9))).item<bool>());
  CHECK(array_equal(norm(arr_one_d, 1.0), array(1 + 2 + 3)).item<bool>());
  CHECK(array_equal(norm(arr_one_d, 0.0), array(3)).item<bool>());

  CHECK(array_equal(norm(arr_one_d, 2.0, {0}, false), array(sqrt(1 + 4 + 9)))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, 2.0, {0}, false),
            array(
                {sqrt(0 + 3 * 3 + 6 * 6),
                 sqrt(1 + 4 * 4 + 7 * 7),
                 sqrt(2 * 2 + 5 * 5 + 8 * 8)}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_two_d, 2.0, {1}, false),
            array(
                {sqrt(0 + 1 + 2 * 2),
                 sqrt(3 * 3 + 4 * 4 + 5 * 5),
                 sqrt(6 * 6 + 7 * 7 + 8 * 8)}))
            .item<bool>());
  CHECK(array_equal(
            norm(arr_three_d, 2.0, {2}, false),
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
            norm(arr_three_d, 2.0, {1}, false),
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
            norm(arr_three_d, 2.0, {0}, false),
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

  CHECK(allclose(
            norm(arr_three_d, 3.0, {0}),
            array(
                {9.,
                 10.00333222,
                 11.02199456,
                 12.06217728,
                 13.12502645,
                 14.2094363,
                 15.31340617,
                 16.43469751,
                 17.57113899},
                {3, 3}))
            .item<bool>());
  CHECK(
      allclose(
          norm(arr_three_d, 3.0, {1}),
          array(
              {6.24025147, 7.41685954, 8.6401226, 18., 19.39257164, 20.7915893},
              {2, 3}))
          .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, 3.0, {2}),
            array(
                {2.08008382,
                 6.,
                 10.23127655,
                 14.5180117,
                 18.82291607,
                 23.13593104},
                {2, 3}))
            .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, 0.0, {0}),
            array({1., 2., 2., 2., 2., 2., 2., 2., 2.}, {3, 3}))
            .item<bool>());
  CHECK(
      allclose(
          norm(arr_three_d, 0.0, {1}), array({2., 3., 3., 3., 3., 3.}, {2, 3}))
          .item<bool>());
  CHECK(
      allclose(
          norm(arr_three_d, 0.0, {2}), array({2., 3., 3., 3., 3., 3.}, {2, 3}))
          .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, 1.0, {0}),
            array({9., 11., 13., 15., 17., 19., 21., 23., 25.}, {3, 3}))
            .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, 1.0, {1}),
            array({9., 12., 15., 36., 39., 42.}, {2, 3}))
            .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, 1.0, {2}),
            array({3., 12., 21., 30., 39., 48.}, {2, 3}))
            .item<bool>());

  CHECK(allclose(norm(arr_two_d, 1.0, {0, 1}), array({15.0})).item<bool>());
  CHECK(allclose(norm(arr_two_d, 1.0, {1, 0}), array({21.0})).item<bool>());
  CHECK(allclose(norm(arr_two_d, -1.0, {0, 1}), array({9.0})).item<bool>());
  CHECK(allclose(norm(arr_two_d, -1.0, {1, 0}), array({3.0})).item<bool>());

  CHECK(allclose(norm(arr_two_d, 1.0, {0, 1}, true), array({15.0}, {1, 1}))
            .item<bool>());
  CHECK(allclose(norm(arr_two_d, 1.0, {1, 0}, true), array({21.0}, {1, 1}))
            .item<bool>());
  CHECK(allclose(norm(arr_two_d, -1.0, {0, 1}, true), array({9.0}, {1, 1}))
            .item<bool>());
  CHECK(allclose(norm(arr_two_d, -1.0, {1, 0}, true), array({3.0}, {1, 1}))
            .item<bool>());

  CHECK(array_equal(norm(arr_two_d, -1.0, {-2, -1}, false), array(9.0))
            .item<bool>());
  CHECK(array_equal(norm(arr_two_d, 1.0, {-2, -1}, false), array(15.0))
            .item<bool>());
  //
  CHECK(allclose(norm(arr_three_d, 1.0, {0, 1}), array({21., 23., 25.}))
            .item<bool>());
  CHECK(
      allclose(norm(arr_three_d, 1.0, {1, 2}), array({15., 42.})).item<bool>());
  CHECK(allclose(norm(arr_three_d, -1.0, {0, 1}), array({9., 11., 13.}))
            .item<bool>());
  CHECK(
      allclose(norm(arr_three_d, -1.0, {1, 2}), array({9., 36.})).item<bool>());
  CHECK(allclose(norm(arr_three_d, -1.0, {1, 0}), array({9., 12., 15.}))
            .item<bool>());
  CHECK(allclose(norm(arr_three_d, -1.0, {2, 1}), array({3, 30})).item<bool>());
  CHECK(allclose(norm(arr_three_d, -1.0, {1, 2}), array({9, 36})).item<bool>());
}

TEST_CASE("[mlx.core.linalg.norm] string ord") {
  array arr_one_d({1, 2, 3});
  array arr_two_d = reshape(arange(9), {3, 3});
  array arr_three_d = reshape(arange(18), {2, 3, 3});

  CHECK(allclose(norm(arr_one_d, "inf", {}), array({3.0})).item<bool>());
  CHECK(allclose(norm(arr_one_d, "-inf", {}), array({1.0})).item<bool>());

  CHECK(allclose(norm(arr_two_d, "f", {0, 1}), array({14.2828568570857}))
            .item<bool>());
  CHECK(allclose(norm(arr_two_d, "fro", {0, 1}), array({14.2828568570857}))
            .item<bool>());
  CHECK(allclose(norm(arr_two_d, "inf", {0, 1}), array({21.0})).item<bool>());
  CHECK(allclose(norm(arr_two_d, "-inf", {0, 1}), array({3.0})).item<bool>());

  CHECK(allclose(
            norm(arr_three_d, "fro", {0, 1}),
            array({22.24859546, 24.31049156, 26.43860813}))
            .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, "fro", {1, 2}), array({14.28285686, 39.7617907}))
            .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, "f", {0, 1}),
            array({22.24859546, 24.31049156, 26.43860813}))
            .item<bool>());
  CHECK(allclose(
            norm(arr_three_d, "f", {1, 0}),
            array({22.24859546, 24.31049156, 26.43860813}))
            .item<bool>());
  CHECK(
      allclose(norm(arr_three_d, "f", {1, 2}), array({14.28285686, 39.7617907}))
          .item<bool>());
  CHECK(
      allclose(norm(arr_three_d, "f", {2, 1}), array({14.28285686, 39.7617907}))
          .item<bool>());
  CHECK(allclose(norm(arr_three_d, "inf", {0, 1}), array({36., 39., 42.}))
            .item<bool>());
  CHECK(allclose(norm(arr_three_d, "inf", {1, 2}), array({21., 48.}))
            .item<bool>());
  CHECK(allclose(norm(arr_three_d, "-inf", {0, 1}), array({9., 12., 15.}))
            .item<bool>());
  CHECK(allclose(norm(arr_three_d, "-inf", {1, 2}), array({3., 30.}))
            .item<bool>());
  CHECK(allclose(norm(arr_three_d, "-inf", {1, 0}), array({9., 11., 13.}))
            .item<bool>());
  CHECK(allclose(norm(arr_three_d, "-inf", {2, 1}), array({9., 36.}))
            .item<bool>());
}