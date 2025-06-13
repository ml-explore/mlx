// Copyright © 2024 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test metal svd basic functionality") {
  // Test basic SVD computation
  array a = array({1.0f, 2.0f, 2.0f, 3.0f}, {2, 2});

  // Test singular values only
  {
    auto s = linalg::svd(a, false);
    CHECK(s.size() == 1);
    CHECK(s[0].shape() == std::vector<int>{2});
    CHECK(s[0].dtype() == float32);
  }

  // Test full SVD
  {
    auto [u, s, vt] = linalg::svd(a, true);
    CHECK(u.shape() == std::vector<int>{2, 2});
    CHECK(s.shape() == std::vector<int>{2});
    CHECK(vt.shape() == std::vector<int>{2, 2});
    CHECK(u.dtype() == float32);
    CHECK(s.dtype() == float32);
    CHECK(vt.dtype() == float32);
  }
}

TEST_CASE("test metal svd input validation") {
  // Test invalid dimensions
  {
    array a = array({1.0f, 2.0f, 3.0f}, {3}); // 1D array
    CHECK_THROWS_AS(linalg::svd(a), std::invalid_argument);
  }

  // Test invalid dtype
  {
    array a = array({1, 2, 2, 3}, {2, 2}); // int32 array
    CHECK_THROWS_AS(linalg::svd(a), std::invalid_argument);
  }

  // Test empty matrix
  {
    array a = array({}, {0, 0});
    CHECK_THROWS_AS(linalg::svd(a), std::invalid_argument);
  }
}

TEST_CASE("test metal svd matrix sizes") {
  // Test various matrix sizes
  std::vector<std::pair<int, int>> sizes = {
      {2, 2},
      {3, 3},
      {4, 4},
      {5, 5},
      {2, 3},
      {3, 2},
      {4, 6},
      {6, 4},
      {8, 8},
      {16, 16},
      {32, 32}};

  for (auto [m, n] : sizes) {
    SUBCASE(("Matrix size " + std::to_string(m) + "x" + std::to_string(n))
                .c_str()) {
      // Create random matrix
      array a = random::normal({m, n}, float32);

      // Test that SVD doesn't crash
      auto [u, s, vt] = linalg::svd(a, true);

      // Check output shapes
      CHECK(u.shape() == std::vector<int>{m, m});
      CHECK(s.shape() == std::vector<int>{std::min(m, n)});
      CHECK(vt.shape() == std::vector<int>{n, n});

      // Check that singular values are non-negative and sorted
      auto s_data = s.data<float>();
      for (int i = 0; i < s.size(); i++) {
        CHECK(s_data[i] >= 0.0f);
        if (i > 0) {
          CHECK(s_data[i] <= s_data[i - 1]); // Descending order
        }
      }
    }
  }
}

TEST_CASE("test metal svd double precision") {
  array a = array({1.0, 2.0, 2.0, 3.0}, {2, 2});
  a = a.astype(float64);

  auto [u, s, vt] = linalg::svd(a, true);

  CHECK(u.dtype() == float64);
  CHECK(s.dtype() == float64);
  CHECK(vt.dtype() == float64);
}

TEST_CASE("test metal svd batch processing") {
  // Test batch of matrices
  array a = random::normal({3, 4, 5}, float32); // 3 matrices of size 4x5

  auto [u, s, vt] = linalg::svd(a, true);

  CHECK(u.shape() == std::vector<int>{3, 4, 4});
  CHECK(s.shape() == std::vector<int>{3, 4});
  CHECK(vt.shape() == std::vector<int>{3, 5, 5});
}

TEST_CASE("test metal svd reconstruction") {
  // Test that U * S * V^T ≈ A
  array a =
      array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, {3, 3});

  auto [u, s, vt] = linalg::svd(a, true);

  // Reconstruct: A_reconstructed = U @ diag(S) @ V^T
  array s_diag = diag(s);
  array reconstructed = matmul(matmul(u, s_diag), vt);

  // Check reconstruction accuracy
  array diff = abs(a - reconstructed);
  float max_error = max(diff).item<float>();
  CHECK(max_error < 1e-5f);
}

TEST_CASE("test metal svd orthogonality") {
  // Test that U and V are orthogonal matrices
  array a = random::normal({4, 4}, float32);

  auto [u, s, vt] = linalg::svd(a, true);

  // Check U^T @ U ≈ I
  array utu = matmul(transpose(u), u);
  array identity = eye(u.shape(0));
  array u_diff = abs(utu - identity);
  float u_max_error = max(u_diff).item<float>();
  CHECK(u_max_error < 1e-4f);

  // Check V^T @ V ≈ I
  array v = transpose(vt);
  array vtv = matmul(transpose(v), v);
  array v_identity = eye(v.shape(0));
  array v_diff = abs(vtv - v_identity);
  float v_max_error = max(v_diff).item<float>();
  CHECK(v_max_error < 1e-4f);
}

TEST_CASE("test metal svd special matrices") {
  // Test identity matrix
  {
    array identity = eye(4);
    auto [u, s, vt] = linalg::svd(identity, true);

    // Singular values should all be 1
    auto s_data = s.data<float>();
    for (int i = 0; i < s.size(); i++) {
      CHECK(abs(s_data[i] - 1.0f) < 1e-6f);
    }
  }

  // Test zero matrix
  {
    array zeros = zeros({3, 3});
    auto [u, s, vt] = linalg::svd(zeros, true);

    // All singular values should be 0
    auto s_data = s.data<float>();
    for (int i = 0; i < s.size(); i++) {
      CHECK(abs(s_data[i]) < 1e-6f);
    }
  }

  // Test diagonal matrix
  {
    array diag_vals = array({3.0f, 2.0f, 1.0f}, {3});
    array diagonal = diag(diag_vals);
    auto [u, s, vt] = linalg::svd(diagonal, true);

    // Singular values should match diagonal values (sorted)
    auto s_data = s.data<float>();
    CHECK(abs(s_data[0] - 3.0f) < 1e-6f);
    CHECK(abs(s_data[1] - 2.0f) < 1e-6f);
    CHECK(abs(s_data[2] - 1.0f) < 1e-6f);
  }
}

TEST_CASE("test metal svd performance characteristics") {
  // Test that larger matrices don't crash and complete in reasonable time
  std::vector<int> sizes = {64, 128, 256};

  for (int size : sizes) {
    SUBCASE(("Performance test " + std::to_string(size) + "x" +
             std::to_string(size))
                .c_str()) {
      array a = random::normal({size, size}, float32);

      auto start = std::chrono::high_resolution_clock::now();
      auto [u, s, vt] = linalg::svd(a, true);
      auto end = std::chrono::high_resolution_clock::now();

      auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

      // Check that computation completed
      CHECK(u.shape() == std::vector<int>{size, size});
      CHECK(s.shape() == std::vector<int>{size});
      CHECK(vt.shape() == std::vector<int>{size, size});

      // Log timing for manual inspection
      MESSAGE(
          "SVD of " << size << "x" << size << " matrix took "
                    << duration.count() << "ms");
    }
  }
}
