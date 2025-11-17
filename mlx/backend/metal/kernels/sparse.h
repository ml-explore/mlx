// Copyright © 2025 Apple Inc.

// Sparse matrix-matrix multiplication: y = A @ B
// where A is sparse (CSR format) and B is a dense matrix
template <typename T>
[[kernel]] void sparse_mm_csr(
    const device int* row_ptr [[buffer(0)]],
    const device int* col_indices [[buffer(1)]],
    const device T* values [[buffer(2)]],
    const device T* dense_matrix [[buffer(3)]],
    device T* output [[buffer(4)]],
    constant int& n_rows [[buffer(5)]],
    constant int& n_cols [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {

  int row = gid.y;
  int col = gid.x;

  if (row >= n_rows || col >= n_cols) {
    return;
  }

  int row_start = row_ptr[row];
  int row_end = row_ptr[row + 1];

  T sum = T(0);
  for (int idx = row_start; idx < row_end; idx++) {
    int k = col_indices[idx];
    sum += values[idx] * dense_matrix[k * n_cols + col];
  }

  output[row * n_cols + col] = sum;
}

// Sparse matrix-vector multiplication: y = A @ x
// where A is sparse (CSR format) and x is a dense vector
template <typename T>
[[kernel]] void sparse_mv_csr(
    const device int* row_ptr [[buffer(0)]],
    const device int* col_indices [[buffer(1)]],
    const device T* values [[buffer(2)]],
    const device T* vector [[buffer(3)]],
    device T* output [[buffer(4)]],
    constant int& n_rows [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {

  int row = gid;
  if (row >= n_rows) {
    return;
  }

  int row_start = row_ptr[row];
  int row_end = row_ptr[row + 1];

  T sum = T(0);
  for (int idx = row_start; idx < row_end; idx++) {
    sum += values[idx] * vector[col_indices[idx]];
  }

  output[row] = sum;
}
