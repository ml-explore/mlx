// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/utils.h"
#include "mlx/extensions.h"
#include "mlx/primitives.h"

namespace mlx::core::ext {

void RoPE::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  assert(inputs.size() == 1);
  assert(outputs.size() == 1);
  auto& in = inputs[0];
  auto& out = outputs[0];

  if (in.ndim() != 3) {
    throw std::runtime_error(
        "[RoPE] Only 3 dimensions are supported (batch x sequence x dims)");
  }
  if (dims_ != in.shape(-1)) {
    throw std::runtime_error("[RoPE] Partial RoPE application not supported");
  }
  if (in.flags().row_contiguous && in.is_donatable()) {
    out.move_shared_buffer(in);
  } else {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }

  auto& s = out.primitive().stream();
  auto& d = metal::device(s.device);
  std::ostringstream kname;
  kname << "rope_" << (traditional_ ? "traditional_" : "") << type_to_name(in);
  auto kernel = d.get_kernel(kname.str());
  auto compute_encoder = d.get_command_encoder(s.index);

  bool donated = in.data_shared_ptr() == nullptr;
  float base = std::log2(base_);
  compute_encoder->setComputePipelineState(kernel);
  set_array_buffer(compute_encoder, donated ? out : in, 0);
  set_array_buffer(compute_encoder, out, 1);
  compute_encoder->setBytes(in.strides().data(), 3 * sizeof(size_t), 2);
  compute_encoder->setBytes(&offset_, sizeof(int), 3);
  compute_encoder->setBytes(&base, sizeof(float), 4);
  compute_encoder->setBytes(&scale_, sizeof(float), 5);

  int dim0 = in.shape(2) / 2;
  int dim1 = in.shape(1);
  int dim2 = in.shape(0);
  auto group_dims = get_block_dims(dim0, dim1, dim2);
  auto grid_dims = MTL::Size(dim0, dim1, dim2);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace mlx::core::ext
