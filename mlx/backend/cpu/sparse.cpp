// Copyright © 2025 Apple Inc.

#include "mlx/backend/cpu/encoder.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void SparseMatmulCSR::eval_cpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& row_ptr = inputs[0];
  auto& col_indices = inputs[1];
  auto& values = inputs[2];
  auto& dense_b = inputs[3];

  auto& encoder = cpu::get_command_encoder(stream());
  encoder.set_input_array(row_ptr);
  encoder.set_input_array(col_indices);
  encoder.set_input_array(values);
  encoder.set_input_array(dense_b);
  encoder.set_output_array(out);

  const int* row_ptr_data = row_ptr.data<int>();
  const int* col_indices_data = col_indices.data<int>();

  int n_rows = n_rows_;
  int n_cols = n_cols_;
  int dense_b_cols = dense_b.shape(1);

  dispatch_float_types(values.dtype(), "sparse_matmul_csr", [&](auto type_tag) {
    using T = MLX_GET_TYPE(type_tag);
    const T* values_data = values.data<T>();
    const T* dense_b_data = dense_b.data<T>();
    T* out_data = out.data<T>();

    encoder.dispatch([row_ptr_data,
                      col_indices_data,
                      values_data,
                      dense_b_data,
                      out_data,
                      n_rows,
                      n_cols,
                      dense_b_cols]() {
      for (int i = 0; i < n_rows * n_cols; i++) {
        out_data[i] = T(0);
      }

      for (int row = 0; row < n_rows; row++) {
        int row_start = row_ptr_data[row];
        int row_end = row_ptr_data[row + 1];

        for (int col = 0; col < n_cols; col++) {
          float sum = 0.0f;

          for (int idx = row_start; idx < row_end; idx++) {
            int k = col_indices_data[idx];
            float a_val = float(values_data[idx]);
            float b_val = float(dense_b_data[k * dense_b_cols + col]);
            sum += a_val * b_val;
          }

          out_data[row * n_cols + col] = T(sum);
        }
      }
    });
  });
}

} // namespace mlx::core
