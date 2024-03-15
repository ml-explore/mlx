// Copyright © 2023-2024 Apple Inc.
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

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
  if (dims_ != in.shape(-1)) {
    throw std::runtime_error("[RoPE] Partial RoPE application not supported");
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);

  size_t strides[3];
  bool donated = false;
  int ndim = in.ndim();
  size_t mat_size = in.shape()[ndim - 2] * in.shape()[ndim - 1];
  if (in.flags().row_contiguous) {
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

  std::ostringstream kname;
  kname << "rope_" << (traditional_ ? "traditional_" : "") << type_to_name(in);
  auto kernel = d.get_kernel(kname.str());
  auto compute_encoder = d.get_command_encoder(s.index);

  float base = std::log2(base_);
  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, donated ? out : in, 0);
  set_array_buffer(compute_encoder, out, 1);
  compute_encoder->setBytes(&strides, 3 * sizeof(size_t), 2);
  compute_encoder->setBytes(&offset_, sizeof(int), 3);
  compute_encoder->setBytes(&base, sizeof(float), 4);
  compute_encoder->setBytes(&scale_, sizeof(float), 5);

  int dim0 = in.shape()[ndim - 1] / 2;
  int dim1 = in.shape()[ndim - 2];
  int dim2 = in.size() / mat_size;
  auto group_dims = get_block_dims(dim0, dim1, dim2);
  auto grid_dims = MTL::Size(dim0, dim1, dim2);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace mlx::core::fast
