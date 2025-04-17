// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernels/cast_op.cuh"
#include "mlx/backend/cuda/reduce/reduce.cuh"

#include <thrust/device_ptr.h>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>

namespace mlx::core {

template <typename... Args>
void cub_all_reduce(cu::CommandEncoder& encoder, Args&&... args) {
  // Allocate temporary storage.
  size_t size;
  CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Run op.
  CHECK_CUDA_ERROR(cub::DeviceReduce::Reduce(temp.data<void>(), size, args...));
}

template <typename... Args>
void cub_segmented_reduce(cu::CommandEncoder& encoder, Args&&... args) {
  // Allocate temporary storage.
  size_t size;
  CHECK_CUDA_ERROR(cub::DeviceSegmentedReduce::Reduce(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Run op.
  CHECK_CUDA_ERROR(
      cub::DeviceSegmentedReduce::Reduce(temp.data<void>(), size, args...));
}

struct MultiplyOp {
  int factor;
  __device__ int operator()(int i) {
    return i * factor;
  }
};

void segmented_reduce(
    cu::CommandEncoder& encoder,
    const array& in,
    array& out,
    Reduce::ReduceType reduce_type,
    const std::vector<int>& axes,
    const ReductionPlan& plan) {
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
      MLX_SWITCH_REDUCE_OPS(reduce_type, OP, {
        using InType = cuda_type_t<CTYPE>;
        using OutType = cu::ReduceResult<OP, InType>::type;
        auto in_iter = cu::make_cast_iterator<OutType>(
            thrust::device_pointer_cast(in.data<InType>()));
        auto out_ptr = thrust::device_pointer_cast(out.data<OutType>());
        auto init = cu::ReduceInit<OP, InType>::value();

        if (plan.type == ContiguousAllReduce) {
          cub_all_reduce(
              encoder, in_iter, out_ptr, in.data_size(), OP(), init, stream);
        } else if (plan.type == ContiguousReduce) {
          auto offsets = thrust::make_transform_iterator(
              thrust::make_counting_iterator(0), MultiplyOp{plan.shape.back()});
          cub_segmented_reduce(
              encoder,
              in_iter,
              out_ptr,
              out.size(),
              offsets,
              offsets + 1,
              OP(),
              init,
              stream);
        } else {
          throw std::runtime_error("Unsupported plan in segmented_reduce.");
        }
      });
    });
  });
}

} // namespace mlx::core
