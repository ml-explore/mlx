// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/reduce/reduce.hpp"
#include "mlx/backend/rocm/device/utils.hpp"
#include "mlx/backend/gpu/copy.h"

#include <hip/hip_runtime.h>
#include <cassert>

namespace mlx::core {

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here.
  assert(!axes_.empty());
  assert(out.size() != in.size());

  auto& s = stream();
  auto& encoder = rocm::get_command_encoder(s);

  if (in.size() == 0) {
    init_reduce(encoder, in, out, reduce_type_);
    return;
  }

  // Size-1 reduction axes are a pure reshape/copy (common after keepdims-style
  // broadcast patterns). Avoid launching col/row_reduce + contig copies.
  bool all_size_one = true;
  for (auto ax : axes_) {
    if (in.shape(ax) != 1) {
      all_size_one = false;
      break;
    }
  }
  if (all_size_one && in.dtype() == out.dtype()) {
    // Donate/copy into out with the reduced shape.
    if (in.flags().row_contiguous && in.size() == out.size()) {
      copy_gpu(in, out, CopyType::Vector, s);
    } else {
      copy_gpu(in, out, CopyType::General, s);
    }
    return;
  }

  // Reduce.
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan. Prefer row_contiguous donation-style skip when the
  // plan already works without a general reduce (cuts copy_gg storms).
  bool broadcasted = false;
  for (int i = 0, j = 0; i < in.ndim() && !broadcasted; i++) {
    if (j < axes_.size() && axes_[j] == i) {
      j++;
    } else {
      broadcasted = in.strides(i) == 0;
    }
  }
  const bool need_contig = plan.type == GeneralReduce || broadcasted ||
      (!in.flags().contiguous && !in.flags().row_contiguous);
  if (need_contig) {
    array in_copy = contiguous_copy_gpu(in, s);
    encoder.add_temporary(in_copy);
    in = in_copy;
    plan = get_reduction_plan(in, axes_);
  }

  if (plan.type == ContiguousAllReduce) {
    all_reduce(encoder, in, out, reduce_type_);
    return;
  }

  if (plan.type == ContiguousReduce || plan.type == GeneralContiguousReduce) {
    row_reduce(encoder, in, out, reduce_type_, axes_, plan);
    return;
  }

  if (plan.type == ContiguousStridedReduce ||
      plan.type == GeneralStridedReduce) {
    col_reduce(encoder, in, out, reduce_type_, axes_, plan);
    return;
  }

  throw std::runtime_error("No plan reached in reduce.");
}

} // namespace mlx::core
