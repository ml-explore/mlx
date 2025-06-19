// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/reduce/reduce.cuh"
#include "mlx/backend/gpu/copy.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include <cassert>

namespace mlx::core {

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Reduce::eval_gpu");
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here.
  assert(!axes_.empty());
  assert(out.size() != in.size());

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);

  if (in.size() == 0) {
    throw std::runtime_error("Should never reach here.");
  }

  // Reduce.
  ReductionPlan plan = get_reduction_plan(in, axes_);

  // If it is a general reduce then copy the input to a contiguous array and
  // recompute the plan.
  if (plan.type == GeneralReduce) {
    array in_copy(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, in_copy, CopyType::General, s);
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
