#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();

  for (auto& out : outputs) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
  }

  std::vector<array> copies;

  auto check_input = [&copies, &s, this](const array& x) -> const array {
    bool no_copy = x.flags().row_contiguous;
    if (!ensure_row_contiguous_ || no_copy) {
      return x;
    } else {
      copies.push_back(array(x.shape(), x.dtype(), nullptr, {}));
      copy_gpu(x, copies.back(), CopyType::General, s);
      return copies.back();
    }
  };
  std::vector<const array> checked_inputs;
  for (const array& in : inputs) {
    checked_inputs.push_back(check_input(in));
  }

  auto& d = metal::device(s.device);
  const auto& lib_name = name_;
  auto lib = d.get_library(lib_name);
  if (lib == nullptr) {
    lib = d.get_library(lib_name, metal::utils() + source_);
  }
  auto kernel = d.get_kernel(name_, lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);
  int index = 0;
  for (const array& in : checked_inputs) {
    compute_encoder.set_input_array(in, index);
    index++;
    if (in.ndim() > 0) {
      int ndim = in.ndim();
      set_vector_bytes(compute_encoder, in.shape(), ndim, index);
      index++;
      set_vector_bytes(compute_encoder, in.strides(), ndim, index);
      index++;
      compute_encoder->setBytes(&ndim, sizeof(int), index);
      index++;
    }
  }
  for (array out : outputs) {
    compute_encoder.set_output_array(out, index);
    index++;
  }

  const auto [tx, ty, tz] = threadgroup_;
  MTL::Size group_dims = MTL::Size(tx, ty, tz);
  const auto [gx, gy, gz] = grid_;
  MTL::Size grid_dims = MTL::Size(gx, gy, gz);
  compute_encoder->dispatchThreads(grid_dims, group_dims);

  if (!copies.empty()) {
    d.get_command_buffer(s.index)->addCompletedHandler(
        [copies](MTL::CommandBuffer*) mutable { copies.clear(); });
  }
}

} // namespace mlx::core::fast
