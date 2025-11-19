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

  uint row_tid = gid.y;
  uint col_vec_idx = gid.x;
  
  // Vector size
  constexpr int BM = 4;
  int col_idx = col_vec_idx * BM;

  if (row_tid >= uint(n_rows) || col_idx >= int(n_cols)) return;

  bool full_vector = (col_idx + BM <= n_cols);
  
  float4 sum = float4(0.0f);
  
  int row_start = row_ptr[row_tid];
  int row_end = row_ptr[row_tid + 1];
  
  if (full_vector) {
      for (int idx = row_start; idx < row_end; idx++) {
          int k = col_indices[idx];
          float val_a = float(values[idx]);
          
          // Vectorized read
          const device packed_vec<T, 4>* src = (const device packed_vec<T, 4>*)(dense_matrix + k * n_cols + col_idx);
          vec<T, 4> val_x_t = *src;
          
          // Convert to float4 for math
          float4 val_x = float4(val_x_t);
          
          sum += val_a * val_x;
      }
      
      // Store
      vec<T, 4> res = vec<T, 4>(sum);
      *((device packed_vec<T, 4>*)(output + row_tid * n_cols + col_idx)) = res;
      
  } else {
      // Tail loop
      for (int idx = row_start; idx < row_end; idx++) {
          int k = col_indices[idx];
          float val_a = float(values[idx]);
          
          for (int i = 0; i < n_cols - col_idx; i++) {
              float val_x = float(dense_matrix[k * n_cols + col_idx + i]);
              sum[i] += val_a * val_x;
          }
      }
      
      for (int i = 0; i < n_cols - col_idx; i++) {
          output[row_tid * n_cols + col_idx + i] = T(sum[i]);
      }
  }
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
