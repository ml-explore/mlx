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
  // Each block processes one row of the sparse matrix
  int row = blockIdx.x;

  if (row >= n_rows) {
    return;
  }

  int row_start = row_ptr[row];
  int row_end = row_ptr[row + 1];

  // Each thread processes multiple columns of the output
  for (int col = threadIdx.x; col < n_cols; col += BLOCK_SIZE) {
    T sum = 0;

    // Iterate through nonzero elements in this row
    for (int idx = row_start; idx < row_end; idx++) {
      int k = col_indices[idx];
      T a_val = values[idx];
      T b_val = dense_b[k * dense_b_cols + col];
      sum += a_val * b_val;
    }

    out[row * n_cols + col] = sum;
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

  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(row_ptr);
  encoder.set_input_array(col_indices);
  encoder.set_input_array(values);
  encoder.set_input_array(dense_b);
  encoder.set_output_array(out);

  int dense_b_cols = dense_b.shape(1);

  // Launch kernel
  dispatch_float_types(values.dtype(), "sparse_matmul_csr", [&](auto type_tag) {
    using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

    constexpr int BLOCK_SIZE = 256;
    dim3 block_dim(BLOCK_SIZE);
    dim3 grid_dim(n_rows_);

    auto kernel = cu::sparse_matmul_csr_kernel<DataType, BLOCK_SIZE>;

    encoder.add_kernel_node(
        kernel,
        grid_dim,
        block_dim,
        0,
        row_ptr.data<int>(),
        col_indices.data<int>(),
        values.data<DataType>(),
        dense_b.data<DataType>(),
        out.data<DataType>(),
        n_rows_,
        n_cols_,
        dense_b_cols);
  });
}

} // namespace mlx::core
