// Copyright © 2025 Apple Inc.

#include "doctest/doctest.h"

#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test sparse matrix-dense matrix multiplication") {
  // Create a simple sparse matrix in CSR format:
  // [[1, 0, 2],
  //  [0, 3, 0],
  //  [4, 0, 5]]
  // CSR: row_ptr = [0, 2, 3, 5]
  //      col_indices = [0, 2, 1, 0, 2]
  //      values = [1, 2, 3, 4, 5]

  auto row_ptr = array({0, 2, 3, 5}, int32);
  auto col_indices = array({0, 2, 1, 0, 2}, int32);
  auto values = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, float32);

  auto dense_a =
      array({1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f}, {3, 3});

  CHECK_EQ(row_ptr.size(), 4);
  CHECK_EQ(col_indices.size(), 5);
  CHECK_EQ(values.size(), 5);

  eval(row_ptr);
  auto row_ptr_data = row_ptr.data<int>();
  CHECK_EQ(row_ptr_data[0], 0);
  CHECK_EQ(row_ptr_data[1], 2);
  CHECK_EQ(row_ptr_data[2], 3);
  CHECK_EQ(row_ptr_data[3], 5);

  auto dense_b = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});

  // Expected result from dense @ dense:
  // [[1, 0, 2],    [[1, 2],      [[11, 14],
  //  [0, 3, 0],  @  [3, 4],   =   [9, 12],
  //  [4, 0, 5]]     [5, 6]]       [29, 38]]
  auto expected = matmul(dense_a, dense_b);

  auto result = sparse_matmul_csr(row_ptr, col_indices, values, dense_b);
  CHECK(allclose(result, expected, 1e-5).item<bool>());

  // Verify result matches expected values
  eval(result);
  auto result_data = result.data<float>();
  CHECK_EQ(result_data[0], 11.0f); // [0,0]
  CHECK_EQ(result_data[1], 14.0f); // [0,1]
  CHECK_EQ(result_data[2], 9.0f); // [1,0]
  CHECK_EQ(result_data[3], 12.0f); // [1,1]
  CHECK_EQ(result_data[4], 29.0f); // [2,0]
  CHECK_EQ(result_data[5], 38.0f); // [2,1]
}

TEST_CASE("test sparse matrix-vector multiplication") {
  // Sparse matrix in CSR format (diagonal):
  // [[2, 0, 0],
  //  [0, 3, 0],
  //  [0, 0, 4]]
  auto row_ptr = array({0, 1, 2, 3}, int32);
  auto col_indices = array({0, 1, 2}, int32);
  auto values = array({2.0f, 3.0f, 4.0f}, float32);

  auto dense_a =
      array({2.0f, 0.0f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f, 0.0f, 4.0f}, {3, 3});

  auto dense_b = array({1.0f, 2.0f, 3.0f}, {3, 1});

  auto expected = matmul(dense_a, dense_b);

  auto result = sparse_matmul_csr(row_ptr, col_indices, values, dense_b);
  CHECK(allclose(result, expected, 1e-5).item<bool>());

  // Verify result values (diagonal matrix times vector)
  eval(result);
  auto result_data = result.data<float>();
  CHECK_EQ(result_data[0], 2.0f); // 2 * 1 = 2
  CHECK_EQ(result_data[1], 6.0f); // 3 * 2 = 6
  CHECK_EQ(result_data[2], 12.0f); // 4 * 3 = 12
}

TEST_CASE("test random sparse matrix") {
  int n_rows = 10;
  int n_cols = 10;
  int dense_cols = 5;

  std::vector<int> row_ptr_vec = {0};
  std::vector<int> col_indices_vec;
  std::vector<float> values_vec;

  for (int i = 0; i < n_rows; i++) {
    int nnz_this_row = 3 + (i % 3);
    for (int j = 0; j < nnz_this_row; j++) {
      col_indices_vec.push_back((i * 3 + j * 2) % n_cols);
      values_vec.push_back(static_cast<float>(i + j + 1));
    }
    row_ptr_vec.push_back(col_indices_vec.size());
  }

  auto row_ptr =
      array(row_ptr_vec.data(), {static_cast<int>(row_ptr_vec.size())}, int32);
  auto col_indices = array(
      col_indices_vec.data(),
      {static_cast<int>(col_indices_vec.size())},
      int32);
  auto values =
      array(values_vec.data(), {static_cast<int>(values_vec.size())}, float32);

  CHECK_EQ(row_ptr.size(), n_rows + 1);
  CHECK(col_indices.size() > 0);
  CHECK_EQ(col_indices.size(), values.size());

  auto dense_b = ones({n_cols, dense_cols});

  auto result = sparse_matmul_csr(row_ptr, col_indices, values, dense_b);
  CHECK_EQ(result.shape(0), n_rows);
  CHECK_EQ(result.shape(1), dense_cols);
}

TEST_CASE("test sparse matmul dtypes") {
  auto row_ptr = array({0, 2, 3, 5}, int32);
  auto col_indices = array({0, 2, 1, 0, 2}, int32);

  auto values_f32 = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, float32);
  auto dense_a_f32 =
      array({1.0f, 0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 4.0f, 0.0f, 5.0f}, {3, 3});
  auto dense_b_f32 = array({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});

  for (auto t : {float16, bfloat16, float32}) {
    auto values = astype(values_f32, t);
    auto dense_a = astype(dense_a_f32, t);
    auto dense_b = astype(dense_b_f32, t);

    auto expected = matmul(dense_a, dense_b);

    auto result = sparse_matmul_csr(row_ptr, col_indices, values, dense_b);
    CHECK(allclose(result, expected, 1e-2).item<bool>());
  }
}

TEST_CASE("test sparse matmul sizes") {
  struct TestCase {
    int m; // sparse matrix rows
    int k; // sparse matrix cols = dense_b rows
    int n; // dense_b cols = output cols
  };

  std::vector<TestCase> test_cases = {
      {1, 1, 1},
      {4, 4, 4},
      {8, 16, 4},
      {16, 8, 32},
      {64, 64, 64},
      {128, 256, 32},
  };

  for (auto& tc : test_cases) {
    std::vector<int> row_ptr_vec = {0};
    std::vector<int> col_indices_vec;
    std::vector<float> values_vec;
    std::vector<float> dense_a_vec(tc.m * tc.k, 0.0f);

    for (int i = 0; i < tc.m; i++) {
      int nnz = std::min(3, tc.k);
      for (int j = 0; j < nnz; j++) {
        int col = (i + j) % tc.k;
        float val = static_cast<float>((i + 1) * (j + 1));
        col_indices_vec.push_back(col);
        values_vec.push_back(val);
        dense_a_vec[i * tc.k + col] += val;
      }
      row_ptr_vec.push_back(col_indices_vec.size());
    }

    auto row_ptr = array(
        row_ptr_vec.data(), {static_cast<int>(row_ptr_vec.size())}, int32);
    auto col_indices = array(
        col_indices_vec.data(),
        {static_cast<int>(col_indices_vec.size())},
        int32);
    auto values = array(
        values_vec.data(), {static_cast<int>(values_vec.size())}, float32);
    auto dense_a = array(dense_a_vec.data(), {tc.m, tc.k}, float32);
    auto dense_b = ones({tc.k, tc.n});

    auto expected = matmul(dense_a, dense_b);

    auto result = sparse_matmul_csr(row_ptr, col_indices, values, dense_b);
    CHECK_EQ(result.shape(0), tc.m);
    CHECK_EQ(result.shape(1), tc.n);
    CHECK(allclose(result, expected, 1e-5).item<bool>());
  }
}
