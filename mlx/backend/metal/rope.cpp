// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

constexpr int n_per_thread = 4;

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& out = outputs[0];

  if (in.ndim() < 3) {
    throw std::runtime_error("[RoPE] Input must have at least 3 dimensions");
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  size_t strides[3];
  size_t out_strides[3];
  bool donated = false;
  int ndim = in.ndim();
  size_t mat_size = in.shape(-2) * in.shape(-1);
  if (dims_ < in.shape(-1)) {
    donated = true;
    auto ctype =
        (in.flags().row_contiguous) ? CopyType::Vector : CopyType::General;
    copy_gpu(in, out, ctype, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  } else if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      donated = true;
      out.move_shared_buffer(in);
    } else {
      out.set_data(allocator::malloc_or_wait(out.nbytes()));
    }
    strides[0] = mat_size;
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else if (ndim == 3) {
    // Handle non-contiguous 3D inputs
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    strides[0] = in.strides()[0];
    strides[1] = in.strides()[1];
    strides[2] = in.strides()[2];
  } else {
    // Copy non-contiguous > 3D inputs into the output and treat
    // input as donated
    donated = true;
    copy_gpu(in, out, CopyType::General, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  }
  out_strides[0] = mat_size;
  out_strides[1] = out.strides()[ndim - 2];
  out_strides[2] = out.strides()[ndim - 1];

  // Special case for inference (single time step and contiguous)
  bool single = in.flags().row_contiguous && (mat_size == in.shape(-1));

  std::ostringstream kname;
  kname << "rope_" << (single ? "single_" : "") << (forward_ ? "" : "vjp_")
        << (traditional_ ? "traditional_" : "") << type_to_name(in);
  auto kernel = d.get_kernel(kname.str());
  auto& compute_encoder = d.get_command_encoder(s.index);

  float base = std::log2(base_);
  compute_encoder->setComputePipelineState(kernel);
  compute_encoder.set_input_array(donated ? out : in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder->setBytes(&offset_, sizeof(int), 2);
  compute_encoder->setBytes(&base, sizeof(float), 3);
  compute_encoder->setBytes(&scale_, sizeof(float), 4);

  size_t n_batch = in.size() / mat_size;
  if (single) {
    compute_encoder->setBytes(&out_strides[1], sizeof(size_t), 5);
    uint32_t dim0 = dims_ / 2;
    auto group_dims = get_block_dims(dim0, n_batch, 1);
    auto grid_dims = MTL::Size(dim0, n_batch, 1);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  } else {
    compute_encoder->setBytes(&strides, 3 * sizeof(size_t), 5);
    compute_encoder->setBytes(&out_strides, 3 * sizeof(size_t), 6);
    compute_encoder->setBytes(&n_batch, sizeof(size_t), 7);
    uint32_t dim0 = dims_ / 2;
    uint32_t dim1 = in.shape(-2);
    uint32_t dim2 = (n_batch + n_per_thread - 1) / n_per_thread;
    auto group_dims = get_block_dims(dim0, dim1, dim2);
    auto grid_dims = MTL::Size(dim0, dim1, dim2);
    compute_encoder.dispatchThreads(grid_dims, group_dims);
  }
}

} // namespace mlx::core::fast
