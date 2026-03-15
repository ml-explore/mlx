// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/cast_op.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>

#include <cassert>

namespace mlx::core {

namespace cu {

template <typename T, int BLOCK_SIZE = 256>
__global__ void sparse_matmul_csr_kernel(
    const int* row_ptr,
    const int* col_indices,
    const T* values,
    const T* dense_b,
    T* out,
    int n_rows,
    int n_cols,
    int dense_b_cols) {
  constexpr int BM = 4;

  int row = blockIdx.y;
  int col_idx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * BM;

  if (row >= n_rows || col_idx >= n_cols)
    return;

  int row_start = row_ptr[row];
  int row_end = row_ptr[row + 1];

  bool full_vector = (col_idx + BM <= n_cols);

  float sum[BM] = {0.0f};

  if (full_vector) {
    for (int idx = row_start; idx < row_end; idx++) {
      int k = col_indices[idx];
      float val_a = static_cast<float>(values[idx]);

      const T* src = dense_b + k * dense_b_cols + col_idx;
#pragma unroll
      for (int i = 0; i < BM; i++) {
        sum[i] += val_a * static_cast<float>(src[i]);
      }
    }

    T* dst = out + row * n_cols + col_idx;
#pragma unroll
    for (int i = 0; i < BM; i++) {
      dst[i] = static_cast<T>(sum[i]);
    }
  } else {
    for (int idx = row_start; idx < row_end; idx++) {
      int k = col_indices[idx];
      float val_a = static_cast<float>(values[idx]);

      for (int i = 0; i < n_cols - col_idx; i++) {
        sum[i] +=
            val_a * static_cast<float>(dense_b[k * dense_b_cols + col_idx + i]);
      }
    }

    for (int i = 0; i < n_cols - col_idx; i++) {
      out[row * n_cols + col_idx + i] = static_cast<T>(sum[i]);
    }
  }
}

} // namespace cu

void SparseMatmulCSR::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("SparseMatmulCSR::eval_gpu");
  assert(inputs.size() == 4);

  const array& row_ptr = inputs[0];
  const array& col_indices = inputs[1];
  const array& values = inputs[2];
  const array& dense_b = inputs[3];

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  out.set_data(cu::malloc_async(out.nbytes(), encoder));

  encoder.set_input_array(row_ptr);
  encoder.set_input_array(col_indices);
  encoder.set_input_array(values);
  encoder.set_input_array(dense_b);
  encoder.set_output_array(out);

  int dense_b_cols = dense_b.shape(1);

  dispatch_float_types(values.dtype(), "sparse_matmul_csr", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    constexpr int BLOCK_SIZE = 256;
    constexpr int BM = 4;
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim((n_cols_ + BM * BLOCK_SIZE - 1) / (BM * BLOCK_SIZE), n_rows_);

    auto kernel = cu::sparse_matmul_csr_kernel<DataType, BLOCK_SIZE>;

    encoder.add_kernel_node(
        kernel,
        grid_dim,
        block_dim,
        0,
        gpu_ptr<int>(row_ptr),
        gpu_ptr<int>(col_indices),
        gpu_ptr<DataType>(values),
        gpu_ptr<DataType>(dense_b),
        gpu_ptr<DataType>(out),
        n_rows_,
        n_cols_,
        dense_b_cols);
  });
}

} // namespace mlx::core
