// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/jit/includes.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/fast_primitives.h"

namespace mlx::core::fast {

void CustomKernel::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();

  std::vector<array> copies;

  for (auto& out : outputs) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    if (init_value_) {
      copies.emplace_back(init_value_.value(), out.dtype());
      fill_gpu(copies.back(), out, s);
    }
  }

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
  std::vector<array> checked_inputs;
  for (const array& in : inputs) {
    checked_inputs.push_back(check_input(in));
  }

  auto& d = metal::device(s.device);
  const auto& lib_name = name_;
  auto lib =
      d.get_library(lib_name, [this] { return metal::utils() + source_; });
  auto kernel = d.get_kernel(name_, lib);
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_compute_pipeline_state(kernel);
  int index = 0;
  for (int i = 0; i < checked_inputs.size(); i++) {
    const array& in = checked_inputs[i];
    auto& shape_info = shape_infos_[i];
    compute_encoder.set_input_array(in, index);
    index++;
    if (in.ndim() > 0) {
      int ndim = in.ndim();
      if (shape_info.shape) {
        compute_encoder.set_vector_bytes(in.shape(), ndim, index);
        index++;
      }
      if (shape_info.strides) {
        compute_encoder.set_vector_bytes(in.strides(), ndim, index);
        index++;
      }
      if (shape_info.ndim) {
        compute_encoder.set_bytes(ndim, index);
        index++;
      }
    }
  }
  for (auto& out : outputs) {
    compute_encoder.set_output_array(out, index);
    index++;
  }

  const auto [tx, ty, tz] = threadgroup_;
  MTL::Size group_dims = MTL::Size(tx, ty, tz);
  const auto [gx, gy, gz] = grid_;
  MTL::Size grid_dims = MTL::Size(gx, gy, gz);
  compute_encoder.dispatchThreads(grid_dims, group_dims);

  d.add_temporaries(std::move(copies), s.index);
}

} // namespace mlx::core::fast
