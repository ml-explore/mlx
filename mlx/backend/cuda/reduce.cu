// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/reduce.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/iterators/cast_iterator.cuh"
#include "mlx/backend/cuda/kernels/reduce_ops.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>

#include <cassert>

namespace mlx::core {

namespace cu {

#define MLX_FORALL_REDUCE_TYPES(_, ...) \
  _(And, __VA_ARGS__)                   \
  _(Or, __VA_ARGS__)                    \
  _(Sum, __VA_ARGS__)                   \
  _(Prod, __VA_ARGS__)                  \
  _(Max, __VA_ARGS__)                   \
  _(Min, __VA_ARGS__)

#define MLX_SWITCH_CASE_REDUCE_TYPE(TYPE, OP, ...) \
  case Reduce::TYPE: {                             \
    using OP = cu::TYPE;                           \
    __VA_ARGS__;                                   \
    break;                                         \
  }

#define MLX_SWITCH_REDUCE_TYPES(TYPE, OP, ...)                            \
  switch (TYPE) {                                                         \
    MLX_FORALL_REDUCE_TYPES(MLX_SWITCH_CASE_REDUCE_TYPE, OP, __VA_ARGS__) \
  }

template <typename Op>
constexpr const char* get_reduce_op_name() {
#define SPECIALIZE_reduce_name(TYPE, ...) \
  if constexpr (std::is_same_v<Op, TYPE>) \
    return #TYPE;
  MLX_FORALL_REDUCE_TYPES(SPECIALIZE_reduce_name, Op)
#undef SPECIALIZE_reduce_name
  return "(unknown reduce op)";
}

template <typename Op, typename T>
constexpr bool supports_reduce_op() {
  if (std::is_same_v<Op, And> || std::is_same_v<Op, Or>) {
    // TODO: Make and/or work for complex number.
    return !std::is_same_v<T, complex64_t>;
  }
  return true;
}

template <typename... Args>
void all_reduce(CommandEncoder& encoder, Args&&... args) {
  // Allocate temporary storage.
  size_t size;
  CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Run op.
  CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(temp.data<void>(), size, args...));
}

template <typename... Args>
void segmented_reduce(CommandEncoder& encoder, Args&&... args) {
  // Allocate temporary storage.
  size_t size;
  CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Reduce(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Run op.
  CHECK_CUDA_ERROR(
      cub::DeviceSegmentedReduce::Reduce(temp.data<void>(), size, args...));
}

} // namespace cu

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Reduce::eval_gpu");
  assert(inputs.size() == 1);
  array in = inputs[0];

  // Make sure no identity reductions trickle down here
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
        MLX_SWITCH_REDUCE_TYPES(reduce_type_, OP, {
          if constexpr (cu::supports_reduce_op<OP, CTYPE>()) {
            using InType = cuda_type_t<CTYPE>;
            using OutType = cu::ReduceResult<OP, InType>::type;
            thrust::fill_n(
                cu::thrust_policy(stream),
                thrust::device_pointer_cast(out.data<OutType>()),
                out.data_size(),
                cu::ReduceInit<OP, InType>::value);
          } else {
            throw std::runtime_error(fmt::format(
                "Can not do reduce init op on dtype {}.",
                dtype_to_string(in.dtype())));
          }
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

  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_TYPES(reduce_type_, OP, {
        if constexpr (cu::supports_reduce_op<OP, CTYPE>()) {
          using InType = cuda_type_t<CTYPE>;
          using OutType = cu::ReduceResult<OP, InType>::type;
          auto in_iter = cu::make_cast_iterator<OutType>(
              thrust::device_pointer_cast(in.data<InType>()));
          auto out_ptr = thrust::device_pointer_cast(out.data<OutType>());
          auto init = cu::ReduceInit<OP, InType>::value;
          if (plan.type == ContiguousAllReduce) {
            cu::all_reduce(
                encoder, in_iter, out_ptr, in.data_size(), OP(), init, stream);
            return;
          }

          if (plan.type == ContiguousReduce && plan.shape.size() == 1) {
            auto offsets = thrust::make_transform_iterator(
                thrust::make_counting_iterator(0),
                [reduction_size = plan.shape.back()] __device__(int i) {
                  return i * reduction_size;
                });
            cu::segmented_reduce(
                encoder,
                in_iter,
                out_ptr,
                out.size(),
                offsets,
                offsets + 1,
                OP(),
                init,
                stream);
            return;
          }

          throw std::runtime_error(
              "Unimplemented reduce layout in CUDA backend.");
        } else {
          throw std::runtime_error(fmt::format(
              "Can not do reduce op {} on dtype {}.",
              cu::get_reduce_op_name<OP>(),
              dtype_to_string(in.dtype())));
        }
      });
    });
  });
}

} // namespace mlx::core
