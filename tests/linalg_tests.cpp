// Copyright © 2023-2024 Apple Inc.

#include "doctest/doctest.h"

#include <cmath>

#include "mlx/mlx.h"
#include "mlx/ops.h"

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
  CHECK_EQ(norm(x, 1.0, std::vector<int>{0, 1}, true).shape(), Shape{1, 1});
  CHECK_EQ(norm(x, 1.0, std::vector<int>{1, 0}, true).shape(), Shape{1, 1});
  CHECK_EQ(norm(x, -1.0, std::vector<int>{0, 1}, true).shape(), Shape{1, 1});
  CHECK_EQ(norm(x, -1.0, std::vector<int>{1, 0}, true).shape(), Shape{1, 1});

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

TEST_CASE("test QR factorization") {
  // 0D and 1D throw
  CHECK_THROWS(linalg::qr(array(0.0)));
  CHECK_THROWS(linalg::qr(array({0.0, 1.0})));

  // Unsupported types throw
  CHECK_THROWS(linalg::qr(array({0, 1}, {1, 2})));

  array A = array({2., 3., 1., 2.}, {2, 2});
  auto [Q, R] = linalg::qr(A, Device::cpu);
  auto out = matmul(Q, R);
  CHECK(allclose(out, A).item<bool>());
  out = matmul(Q, Q);
  CHECK(allclose(out, eye(2), 1e-5, 1e-7).item<bool>());
  CHECK(allclose(tril(R, -1), zeros_like(R)).item<bool>());
  CHECK_EQ(Q.dtype(), float32);
  CHECK_EQ(R.dtype(), float32);
}

TEST_CASE("test SVD factorization") {
  // 0D and 1D throw
  CHECK_THROWS(linalg::svd(array(0.0)));
  CHECK_THROWS(linalg::svd(array({0.0, 1.0})));

  // Unsupported types throw
  CHECK_THROWS(linalg::svd(array({0, 1}, {1, 2})));

  const auto prng_key = random::key(42);
  const auto A = mlx::core::random::normal({5, 4}, prng_key);
  const auto outs = linalg::svd(A, Device::cpu);
  CHECK_EQ(outs.size(), 3);

  const auto& U = outs[0];
  const auto& S = outs[1];
  const auto& Vt = outs[2];

  CHECK_EQ(U.shape(), Shape{5, 5});
  CHECK_EQ(S.shape(), Shape{4});
  CHECK_EQ(Vt.shape(), Shape{4, 4});

  const auto U_slice = slice(U, {0, 0}, {U.shape(0), S.shape(0)});

  const auto A_again = matmul(matmul(U_slice, diag(S)), Vt);

  CHECK(
      allclose(A_again, A, /* rtol = */ 1e-4, /* atol = */ 1e-4).item<bool>());
  CHECK_EQ(U.dtype(), float32);
  CHECK_EQ(S.dtype(), float32);
  CHECK_EQ(Vt.dtype(), float32);
}

TEST_CASE("test matrix inversion") {
  // 0D and 1D throw
  CHECK_THROWS(linalg::inv(array(0.0), Device::cpu));
  CHECK_THROWS(linalg::inv(array({0.0, 1.0}), Device::cpu));

  // Unsupported types throw
  CHECK_THROWS(linalg::inv(array({0, 1}, {1, 2}), Device::cpu));

  // Non-square throws.
  CHECK_THROWS(linalg::inv(array({1, 2, 3, 4, 5, 6}, {2, 3}), Device::cpu));

  const auto prng_key = random::key(42);
  const auto A = random::normal({5, 5}, prng_key);
  const auto A_inv = linalg::inv(A, Device::cpu);
  const auto identity = eye(A.shape(0));

  CHECK(allclose(matmul(A, A_inv), identity, /* rtol = */ 0, /* atol = */ 1e-6)
            .item<bool>());
  CHECK(allclose(matmul(A_inv, A), identity, /* rtol = */ 0, /* atol = */ 1e-6)
            .item<bool>());
}

TEST_CASE("test matrix cholesky") {
  // 0D and 1D throw
  CHECK_THROWS(linalg::cholesky(array(0.0), /* upper = */ false, Device::cpu));
  CHECK_THROWS(
      linalg::cholesky(array({0.0, 1.0}), /* upper = */ false, Device::cpu));

  // Unsupported types throw
  CHECK_THROWS(linalg::cholesky(
      array({0, 1}, {1, 2}), /* upper = */ false, Device::cpu));

  // Non-square throws.
  CHECK_THROWS(linalg::cholesky(
      array({1, 2, 3, 4, 5, 6}, {2, 3}), /* upper = */ false, Device::cpu));

  const auto prng_key = random::key(220398);
  const auto sqrtA = random::normal({5, 5}, prng_key);
  const auto A = matmul(sqrtA, transpose(sqrtA));
  const auto L = linalg::cholesky(A, /* upper = */ false, Device::cpu);
  const auto U = linalg::cholesky(A, /* upper = */ true, Device::cpu);

  CHECK(allclose(matmul(L, transpose(L)), A, /* rtol = */ 0, /* atol = */ 1e-6)
            .item<bool>());
  CHECK(allclose(matmul(transpose(U), U), A, /* rtol = */ 0, /* atol = */ 1e-6)
            .item<bool>());
}

TEST_CASE("test matrix pseudo-inverse") {
  // 0D and 1D throw
  CHECK_THROWS(linalg::pinv(array(0.0), Device::cpu));
  CHECK_THROWS(linalg::pinv(array({0.0, 1.0}), Device::cpu));

  // Unsupported types throw
  CHECK_THROWS(linalg::pinv(array({0, 1}, {1, 2}), Device::cpu));

  { // Square m == n
    const auto A = array({1.0, 2.0, 3.0, 4.0}, {2, 2});
    const auto A_pinv = linalg::pinv(A, Device::cpu);
    const auto A_again = matmul(matmul(A, A_pinv), A);
    CHECK(allclose(A_again, A).item<bool>());
    const auto A_pinv_again = matmul(matmul(A_pinv, A), A_pinv);
    CHECK(allclose(A_pinv_again, A_pinv).item<bool>());
  }
  { // Rectangular matrix m < n
    const auto prng_key = random::key(42);
    const auto A = random::normal({4, 5}, prng_key);
    const auto A_pinv = linalg::pinv(A, Device::cpu);
    const auto zeros = zeros_like(A_pinv, Device::cpu);
    CHECK_FALSE(allclose(zeros, A_pinv, /* rtol = */ 0, /* atol = */ 1e-6)
                    .item<bool>());
    const auto A_again = matmul(matmul(A, A_pinv), A);
    CHECK(allclose(A_again, A).item<bool>());
    const auto A_pinv_again = matmul(matmul(A_pinv, A), A_pinv);
    CHECK(allclose(A_pinv_again, A_pinv).item<bool>());
  }
  { // Rectangular matrix m > n
    const auto prng_key = random::key(10);
    const auto A = random::normal({6, 5}, prng_key);
    const auto A_pinv = linalg::pinv(A, Device::cpu);
    const auto zeros2 = zeros_like(A_pinv, Device::cpu);
    CHECK_FALSE(allclose(zeros2, A_pinv, /* rtol = */ 0, /* atol = */ 1e-6)
                    .item<bool>());
    const auto A_again = matmul(matmul(A, A_pinv), A);
    CHECK(allclose(A_again, A).item<bool>());
    const auto A_pinv_again = matmul(matmul(A_pinv, A), A_pinv);
    CHECK(allclose(A_pinv_again, A_pinv).item<bool>());
  }
}

TEST_CASE("test cross product") {
  using namespace mlx::core::linalg;

  // Test for vectors of length 3
  array a = array({1.0, 2.0, 3.0});
  array b = array({4.0, 5.0, 6.0});

  array expected = array(
      {2.0 * 6.0 - 3.0 * 5.0, 3.0 * 4.0 - 1.0 * 6.0, 1.0 * 5.0 - 2.0 * 4.0});

  array result = cross(a, b);
  CHECK(allclose(result, expected).item<bool>());

  // Test for vectors of length 3 with negative values
  a = array({-1.0, -2.0, -3.0});
  b = array({4.0, -5.0, 6.0});

  expected = array(
      {-2.0 * 6.0 - (-3.0 * -5.0),
       -3.0 * 4.0 - (-1.0 * 6.0),
       -1.0 * -5.0 - (-2.0 * 4.0)});

  result = cross(a, b);
  CHECK(allclose(result, expected).item<bool>());

  // Test for incorrect vector size (should throw)
  b = array({1.0, 2.0});
  expected = array(
      {-2.0 * 0.0 - (-3.0 * 2.0),
       -3.0 * 1.0 - (-1.0 * 0.0),
       -1.0 * 2.0 - (-2.0 * 1.0)});

  result = cross(a, b);
  CHECK(allclose(result, expected).item<bool>());

  // Test for vectors of length 3 with integer values
  a = array({1, 2, 3});
  b = array({4, 5, 6});

  expected = array({2 * 6 - 3 * 5, 3 * 4 - 1 * 6, 1 * 5 - 2 * 4});

  result = cross(a, b);
  CHECK(allclose(result, expected).item<bool>());
}

TEST_CASE("test matrix eigh") {
  // 0D and 1D throw
  CHECK_THROWS(linalg::eigh(array(0.0)));
  CHECK_THROWS(linalg::eigh(array({0.0, 1.0})));
  CHECK_THROWS(linalg::eigvalsh(array(0.0)));
  CHECK_THROWS(linalg::eigvalsh(array({0.0, 1.0})));

  // Unsupported types throw
  CHECK_THROWS(linalg::eigh(array({0, 1}, {1, 2})));

  // Non-square throws
  CHECK_THROWS(linalg::eigh(array({1, 2, 3, 4, 5, 6}, {2, 3})));

  // Test a simple 2x2 symmetric matrix
  array A = array({1.0, 2.0, 2.0, 4.0}, {2, 2}, float32);
  auto [eigvals, eigvecs] = linalg::eigh(A, "L", Device::cpu);

  // Expected eigenvalues
  array expected_eigvals = array({0.0, 5.0});
  CHECK(allclose(
            eigvals,
            expected_eigvals,
            /* rtol = */ 1e-5,
            /* atol = */ 1e-5)
            .item<bool>());

  // Verify orthogonality of eigenvectors
  CHECK(allclose(
            matmul(eigvecs, transpose(eigvecs)),
            eye(2),
            /* rtol = */ 1e-5,
            /* atol = */ 1e-5)
            .item<bool>());

  // Verify eigendecomposition
  CHECK(allclose(matmul(A, eigvecs), eigvals * eigvecs).item<bool>());
}
