// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <cassert>

namespace mlx::core {

namespace {

template <typename T>
struct ModOp {
  T divisor;
  __device__ T operator()(T x) {
    return x % divisor;
  }
};

struct OffsetTransform {
  int nsort;

  int __device__ operator()(int i) {
    return i * nsort;
  }
};

void gpu_sort(const Stream& s, array in, array& out_, int axis, bool argsort) {
  array out = out_;
  auto& encoder = cu::get_command_encoder(s);
  if (axis < 0) {
    axis += in.ndim();
  }
  int nsort = in.shape(axis);
  int last_dim = in.ndim() - 1;

  // If we are not sorting the innermost dimension of a contiguous array,
  // transpose and make a copy.
  bool is_segmented_sort = in.flags().contiguous && in.strides()[axis] == 1;
  if (!is_segmented_sort) {
    array trans = swapaxes_in_eval(in, axis, last_dim);
    in = contiguous_copy_gpu(trans, s);
    encoder.add_temporary(in);
    out = array(allocator::malloc(out.nbytes()), in.shape(), out.dtype());
    encoder.add_temporary(out);
  } else {
    out.set_data(
        allocator::malloc(in.data_size() * out.itemsize()),
        in.data_size(),
        in.strides(),
        in.flags());
  }

  encoder.set_input_array(in);
  encoder.set_output_array(out);
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    auto& stream = encoder.stream();
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using Type = cuda_type_t<CTYPE>;
      auto offsets = thrust::make_transform_iterator(
          thrust::make_counting_iterator(0), OffsetTransform{nsort});
      if (argsort) {
        // Indices in the sorted dimension.
        array indices(allocator::malloc(out.nbytes()), in.shape(), out.dtype());
        encoder.add_temporary(indices);

        // In argsort though we don't need the result of sorted values, the
        // API requires us to provide an array to store it.
        array discard(allocator::malloc(in.nbytes()), in.shape(), in.dtype());
        encoder.add_temporary(discard);

        size_t size;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr,
            size,
            in.data<Type>(),
            discard.data<Type>(),
            indices.data<uint32_t>(),
            out.data<uint32_t>(),
            in.data_size(),
            in.data_size() / nsort,
            offsets,
            offsets + 1,
            0,
            sizeof(Type) * 8,
            stream));

        array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
        encoder.add_temporary(temp);

        // Start capturing after allocations
        auto capture = encoder.capture_context();
        thrust::transform(
            cu::thrust_policy(stream),
            thrust::counting_iterator<uint32_t>(0),
            thrust::counting_iterator<uint32_t>(indices.data_size()),
            thrust::device_pointer_cast(indices.data<uint32_t>()),
            ModOp<uint32_t>{static_cast<uint32_t>(nsort)});

        CHECK_CUDA_ERROR(cub::DeviceSegmentedRadixSort::SortPairs(
            temp.data<void>(),
            size,
            in.data<Type>(),
            discard.data<Type>(),
            indices.data<uint32_t>(),
            out.data<uint32_t>(),
            in.data_size(),
            in.data_size() / nsort,
            offsets,
            offsets + 1,
            0,
            sizeof(Type) * 8,
            stream));
      } else {
        size_t size;
        CHECK_CUDA_ERROR(cub::DeviceSegmentedRadixSort::SortKeys(
            nullptr,
            size,
            in.data<Type>(),
            out.data<Type>(),
            in.data_size(),
            in.data_size() / nsort,
            offsets,
            offsets + 1,
            0,
            sizeof(Type) * 8,
            stream));

        array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
        encoder.add_temporary(temp);

        // Start capturing after allocations
        auto capture = encoder.capture_context();
        CHECK_CUDA_ERROR(cub::DeviceSegmentedRadixSort::SortKeys(
            temp.data<void>(),
            size,
            in.data<Type>(),
            out.data<Type>(),
            in.data_size(),
            in.data_size() / nsort,
            offsets,
            offsets + 1,
            0,
            sizeof(Type) * 8,
            stream));
      }
    } else {
      throw std::runtime_error(
          "CUDA backend does not support sorting complex numbers");
    }
  });

  if (!is_segmented_sort) {
    // Swap the sorted axis back.
    // TODO: Do in-place transpose instead of using a temporary out array.
    copy_gpu(swapaxes_in_eval(out, axis, last_dim), out_, CopyType::General, s);
  }
}

} // namespace

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("ArgSort::eval_gpu");
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, true);
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Sort::eval_gpu");
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, false);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("ArgPartition::eval_gpu");
  gpu_sort(stream(), inputs[0], out, axis_, true);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  nvtx3::scoped_range r("Partition::eval_gpu");
  gpu_sort(stream(), inputs[0], out, axis_, false);
}

} // namespace mlx::core
