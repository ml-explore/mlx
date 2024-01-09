// Copyright © 2023 Apple Inc.

#include "doctest/doctest.h"

#include <cmath>

#include "mlx/mlx.h"

using namespace mlx::core;
using namespace mlx::core::linalg;

TEST_CASE("[mlx.core.linalg.norm] no ord") {
  // Zero dimensions
  array x(2.0);
  CHECK_EQ(norm(x).item<float>(), 2.0f);
  CHECK_THROWS(norm(x, 0));

  x = array({1, 2, 3});
  float expected = std::sqrt(1 + 4 + 9);
  CHECK_EQ(norm(x).item<float>(), doctest::Approx(expected));
  CHECK_EQ(norm(x, 0, false).item<float>(), doctest::Approx(expected));
  CHECK_EQ(norm(x, -1, false).item<float>(), doctest::Approx(expected));
  CHECK_EQ(norm(x, -1, true).ndim(), 1);
  CHECK_THROWS(norm(x, 1));

  x = reshape(arange(9), {3, 3});
  expected =
      std::sqrt(0 + 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 + 8 * 8);

  CHECK_EQ(norm(x).item<float>(), doctest::Approx(expected));
  CHECK_EQ(
      norm(x, std::vector<int>{0, 1}).item<float>(), doctest::Approx(expected));
  CHECK(allclose(
            norm(x, 0, false),
            array(
                {std::sqrt(0 + 3 * 3 + 6 * 6),
                 std::sqrt(1 + 4 * 4 + 7 * 7),
                 std::sqrt(2 * 2 + 5 * 5 + 8 * 8)}))
            .item<bool>());
  CHECK(allclose(
            norm(x, 1, false),
            array(
                {std::sqrt(0 + 1 + 2 * 2),
                 std::sqrt(3 * 3 + 4 * 4 + 5 * 5),
                 std::sqrt(6 * 6 + 7 * 7 + 8 * 8)}))
            .item<bool>());

  x = reshape(arange(18), {2, 3, 3});
  CHECK(allclose(
            norm(x, 2, false),
            array(
                {
                    std::sqrt(0 + 1 + 2 * 2),
                    std::sqrt(3 * 3 + 4 * 4 + 5 * 5),
                    std::sqrt(6 * 6 + 7 * 7 + 8 * 8),
                    std::sqrt(9 * 9 + 10 * 10 + 11 * 11),
                    std::sqrt(12 * 12 + 13 * 13 + 14 * 14),
                    std::sqrt(15 * 15 + 16 * 16 + 17 * 17),
                },
                {2, 3}))
            .item<bool>());
  CHECK(allclose(
            norm(x, std::vector<int>{1, 2}, false),
            array(
                {std::sqrt(
                     0 + 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 +
                     8 * 8),
                 std::sqrt(
                     9 * 9 + 10 * 10 + 11 * 11 + 12 * 12 + 13 * 13 + 14 * 14 +
                     15 * 15 + 16 * 16 + 17 * 17)},
                {2}))
            .item<bool>());
  CHECK_THROWS(norm(x, std::vector<int>{0, 1, 2}));
}

TEST_CASE("[mlx.core.linalg.norm] double ord") {
  CHECK_THROWS(norm(array(0), 2.0));

  array x({1, 2, 3});

  float expected = std::sqrt(1 + 4 + 9);
  CHECK_EQ(norm(x, 2.0).item<float>(), doctest::Approx(expected));
  CHECK_EQ(norm(x, 2.0, 0).item<float>(), doctest::Approx(expected));
  CHECK_THROWS(norm(x, 2.0, 1));

  expected = 1 + 2 + 3;
  CHECK_EQ(norm(x, 1.0).item<float>(), doctest::Approx(expected));

  expected = 3;
  CHECK_EQ(norm(x, 0.0).item<float>(), doctest::Approx(expected));

  expected = 3;
  CHECK_EQ(
      norm(x, std::numeric_limits<double>::infinity()).item<float>(),
      doctest::Approx(expected));

  expected = 1;
  CHECK_EQ(
      norm(x, -std::numeric_limits<double>::infinity()).item<float>(),
      doctest::Approx(expected));

  x = reshape(arange(9), {3, 3});

  CHECK(allclose(
            norm(x, 2.0, 0, false),
            array(
                {std::sqrt(0 + 3 * 3 + 6 * 6),
                 std::sqrt(1 + 4 * 4 + 7 * 7),
                 std::sqrt(2 * 2 + 5 * 5 + 8 * 8)}))
            .item<bool>());
  CHECK(allclose(
            norm(x, 2.0, 1, false),
            array(
                {sqrt(0 + 1 + 2 * 2),
                 sqrt(3 * 3 + 4 * 4 + 5 * 5),
                 sqrt(6 * 6 + 7 * 7 + 8 * 8)}))
            .item<bool>());

  CHECK_EQ(
      norm(x, 1.0, std::vector<int>{0, 1}).item<float>(),
      doctest::Approx(15.0));
  CHECK_EQ(
      norm(x, 1.0, std::vector<int>{1, 0}).item<float>(),
      doctest::Approx(21.0));
  CHECK_EQ(
      norm(x, -1.0, std::vector<int>{0, 1}).item<float>(),
      doctest::Approx(9.0));
  CHECK_EQ(
      norm(x, -1.0, std::vector<int>{1, 0}).item<float>(),
      doctest::Approx(3.0));
  CHECK_EQ(
      norm(x, 1.0, std::vector<int>{0, 1}, true).shape(),
      std::vector<int>{1, 1});
  CHECK_EQ(
      norm(x, 1.0, std::vector<int>{1, 0}, true).shape(),
      std::vector<int>{1, 1});
  CHECK_EQ(
      norm(x, -1.0, std::vector<int>{0, 1}, true).shape(),
      std::vector<int>{1, 1});
  CHECK_EQ(
      norm(x, -1.0, std::vector<int>{1, 0}, true).shape(),
      std::vector<int>{1, 1});

  CHECK_EQ(
      norm(x, -1.0, std::vector<int>{-2, -1}, false).item<float>(),
      doctest::Approx(9.0));
  CHECK_EQ(
      norm(x, 1.0, std::vector<int>{-2, -1}, false).item<float>(),
      doctest::Approx(15.0));

  x = reshape(arange(18), {2, 3, 3});
  CHECK_THROWS(norm(x, 2.0, std::vector{0, 1, 2}));
  CHECK(allclose(
            norm(x, 3.0, 0),
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
  CHECK(allclose(
            norm(x, 3.0, 2),
            array(
                {2.08008382,
                 6.,
                 10.23127655,
                 14.5180117,
                 18.82291607,
                 23.13593104},
                {2, 3}))
            .item<bool>());
  CHECK(
      allclose(
          norm(x, 0.0, 0), array({1., 2., 2., 2., 2., 2., 2., 2., 2.}, {3, 3}))
          .item<bool>());
  CHECK(allclose(norm(x, 0.0, 1), array({2., 3., 3., 3., 3., 3.}, {2, 3}))
            .item<bool>());
  CHECK(allclose(norm(x, 0.0, 2), array({2., 3., 3., 3., 3., 3.}, {2, 3}))
            .item<bool>());
  CHECK(allclose(
            norm(x, 1.0, 0),
            array({9., 11., 13., 15., 17., 19., 21., 23., 25.}, {3, 3}))
            .item<bool>());
  CHECK(allclose(norm(x, 1.0, 1), array({9., 12., 15., 36., 39., 42.}, {2, 3}))
            .item<bool>());
  CHECK(allclose(norm(x, 1.0, 2), array({3., 12., 21., 30., 39., 48.}, {2, 3}))
            .item<bool>());

  CHECK(allclose(norm(x, 1.0, std::vector<int>{0, 1}), array({21., 23., 25.}))
            .item<bool>());
  CHECK(allclose(norm(x, 1.0, std::vector<int>{1, 2}), array({15., 42.}))
            .item<bool>());
  CHECK(allclose(norm(x, -1.0, std::vector<int>{0, 1}), array({9., 11., 13.}))
            .item<bool>());
  CHECK(allclose(norm(x, -1.0, std::vector<int>{1, 2}), array({9., 36.}))
            .item<bool>());
  CHECK(allclose(norm(x, -1.0, std::vector<int>{1, 0}), array({9., 12., 15.}))
            .item<bool>());
  CHECK(allclose(norm(x, -1.0, std::vector<int>{2, 1}), array({3, 30}))
            .item<bool>());
  CHECK(allclose(norm(x, -1.0, std::vector<int>{1, 2}), array({9, 36}))
            .item<bool>());
}

TEST_CASE("[mlx.core.linalg.norm] string ord") {
  array x({1, 2, 3});
  CHECK_THROWS(norm(x, "fro"));

  x = reshape(arange(9), {3, 3});
  CHECK_THROWS(norm(x, "bad ord"));

  CHECK_EQ(
      norm(x, "f", std::vector<int>{0, 1}).item<float>(),
      doctest::Approx(14.2828568570857));
  CHECK_EQ(
      norm(x, "fro", std::vector<int>{0, 1}).item<float>(),
      doctest::Approx(14.2828568570857));

  x = reshape(arange(18), {2, 3, 3});
  CHECK(allclose(
            norm(x, "fro", std::vector<int>{0, 1}),
            array({22.24859546, 24.31049156, 26.43860813}))
            .item<bool>());
  CHECK(allclose(
            norm(x, "fro", std::vector<int>{1, 2}),
            array({14.28285686, 39.7617907}))
            .item<bool>());
  CHECK(allclose(
            norm(x, "f", std::vector<int>{0, 1}),
            array({22.24859546, 24.31049156, 26.43860813}))
            .item<bool>());
  CHECK(allclose(
            norm(x, "f", std::vector<int>{1, 0}),
            array({22.24859546, 24.31049156, 26.43860813}))
            .item<bool>());
  CHECK(allclose(
            norm(x, "f", std::vector<int>{1, 2}),
            array({14.28285686, 39.7617907}))
            .item<bool>());
  CHECK(allclose(
            norm(x, "f", std::vector<int>{2, 1}),
            array({14.28285686, 39.7617907}))
            .item<bool>());
}
