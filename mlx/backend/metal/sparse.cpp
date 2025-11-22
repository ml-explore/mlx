// Copyright © 2025 Apple Inc.

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

void SparseMatmulCSR::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 4);

  auto& row_ptr = inputs[0];
  auto& col_indices = inputs[1];
  auto& values = inputs[2];
  auto& dense_b = inputs[3];

  auto& s = stream();
  auto& d = metal::device(s.device);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& compute_encoder = d.get_command_encoder(s.index);

  std::string kernel_name = "sparse_mm_csr_" + type_to_name(values);
  auto kernel = get_sparse_kernel(d, kernel_name);

  MTL::Size grid_dims = MTL::Size((n_cols_ + 3) / 4, n_rows_, 1);
  MTL::Size group_dims = MTL::Size(32, 1, 1);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(row_ptr, 0);
  compute_encoder.set_input_array(col_indices, 1);
  compute_encoder.set_input_array(values, 2);
  compute_encoder.set_input_array(dense_b, 3);
  compute_encoder.set_output_array(out, 4);
  compute_encoder.set_bytes(n_rows_, 5);
  compute_encoder.set_bytes(n_cols_, 6);
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core
