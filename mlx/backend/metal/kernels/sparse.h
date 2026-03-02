// Copyright © 2025 Apple Inc.

constant constexpr int BLOCK_SIZE = 64;

template <typename T>
[[kernel]] void sparse_mm_csr(
    const device int* row_ptr [[buffer(0)]],
    const device int* col_indices [[buffer(1)]],
    const device T* values [[buffer(2)]],
    const device T* dense_matrix [[buffer(3)]],
    device T* output [[buffer(4)]],
    constant int& n_rows [[buffer(5)]],
    constant int& n_cols [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  uint row_tid = gid.y;
  uint col_vec_idx = gid.x;

  constexpr int BM = 4;
  int col_idx = col_vec_idx * BM;

  threadgroup int shared_cols[BLOCK_SIZE];
  threadgroup float shared_vals[BLOCK_SIZE];

  // All threads in threadgroup share the same row, so safe to return early
  if (row_tid >= uint(n_rows))
    return;

  // Use flags instead of early return to maintain barrier synchronization
  bool valid_col = (col_idx < n_cols);
  bool full_vector = valid_col && (col_idx + BM <= n_cols);

  float4 sum = float4(0.0f);

  int row_start = row_ptr[row_tid];
  int row_end = row_ptr[row_tid + 1];
  int nnz = row_end - row_start;

  for (int nz_base = 0; nz_base < nnz; nz_base += BLOCK_SIZE) {
    int chunk_size = min(BLOCK_SIZE, nnz - nz_base);

    if (int(tid) < chunk_size) {
      shared_cols[tid] = col_indices[row_start + nz_base + tid];
      shared_vals[tid] = float(values[row_start + nz_base + tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (full_vector) {
      for (int i = 0; i < chunk_size; i++) {
        int k = shared_cols[i];
        float val_a = shared_vals[i];

        const device packed_vec<T, 4>* src =
            (const device packed_vec<T, 4>*)(dense_matrix + k * n_cols +
                                             col_idx);
        vec<T, 4> val_x_t = *src;
        float4 val_x = float4(val_x_t);

        sum += val_a * val_x;
      }
    } else if (valid_col) {
      for (int i = 0; i < chunk_size; i++) {
        int k = shared_cols[i];
        float val_a = shared_vals[i];

        for (int j = 0; j < n_cols - col_idx; j++) {
          float val_x = float(dense_matrix[k * n_cols + col_idx + j]);
          sum[j] += val_a * val_x;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (full_vector) {
    vec<T, 4> res = vec<T, 4>(sum);
    *((device packed_vec<T, 4>*)(output + row_tid * n_cols + col_idx)) = res;
  } else if (valid_col) {
    for (int i = 0; i < n_cols - col_idx; i++) {
      output[row_tid * n_cols + col_idx + i] = T(sum[i]);
    }
  }
}
