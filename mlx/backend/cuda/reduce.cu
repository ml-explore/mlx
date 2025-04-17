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

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  // Fill out with init value.
  if (in.size() == 0) {
    encoder.launch_kernel([&](cudaStream_t stream) {
      MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
        MLX_SWITCH_REDUCE_OPS(reduce_type_, OP, {
          using InType = cuda_type_t<CTYPE>;
          using OutType = cu::ReduceResult<OP, InType>::type;
          thrust::fill_n(
              cu::thrust_policy(stream),
              thrust::device_pointer_cast(out.data<OutType>()),
              out.data_size(),
              cu::ReduceInit<OP, InType>::value());
        });
      });
    });
    return;
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

  if ((plan.type == ContiguousAllReduce) ||
      (plan.type == ContiguousReduce && plan.shape.size() == 1)) {
    segmented_reduce(encoder, in, out, reduce_type_, axes_, plan);
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
