// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <nvtx3/nvtx3.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <cub/device/device_segmented_sort.cuh>

#include <cassert>
#include <numeric>

namespace mlx::core {

namespace {

template <typename T>
struct ModOp {
  T divisor;
  __device__ T operator()(T x) {
    return x % divisor;
  }
};

// We can not use any op in eval, make an utility.
array swapaxes_in_eval(const array& in, int axis1, int axis2) {
  std::vector<int> axes(in.ndim());
  std::iota(axes.begin(), axes.end(), 0);
  std::swap(axes[axis1], axes[axis2]);
  // TODO: Share the code with Transpose::eval.
  Shape shape(axes.size());
  Strides strides(in.ndim());
  for (size_t ax = 0; ax < axes.size(); ++ax) {
    shape[ax] = in.shape()[axes[ax]];
    strides[ax] = in.strides()[axes[ax]];
  }
  auto flags = in.flags();
  if (flags.contiguous) {
    auto [_, row_contiguous, col_contiguous] = check_contiguity(shape, strides);
    flags.row_contiguous = row_contiguous;
    flags.col_contiguous = col_contiguous;
  }
  array out(shape, in.dtype(), nullptr, {});
  out.copy_shared_buffer(in, strides, flags, in.data_size());
  return out;
}

template <typename... Args>
void segmented_sort_pairs(cu::CommandEncoder& encoder, Args&&... args) {
  // Allocate temporary storage.
  size_t size;
  CHECK_CUDA_ERROR(
      cub::DeviceSegmentedSort::StableSortPairs(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Run op.
  CHECK_CUDA_ERROR(cub::DeviceSegmentedSort::StableSortPairs(
      temp.data<void>(), size, args...));
}

template <typename... Args>
void segmented_sort(cu::CommandEncoder& encoder, Args&&... args) {
  // Allocate temporary storage.
  size_t size;
  CHECK_CUDA_ERROR(
      cub::DeviceSegmentedSort::StableSortKeys(nullptr, size, args...));
  array temp(allocator::malloc(size), {static_cast<int>(size)}, uint8);
  encoder.add_temporary(temp);
  // Run op.
  CHECK_CUDA_ERROR(cub::DeviceSegmentedSort::StableSortKeys(
      temp.data<void>(), size, args...));
}

void gpu_sort(const Stream& s, array in, array& out_, int axis, bool argsort) {
  array out = out_;
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);

  if (axis < 0) {
    axis += in.ndim();
  }
  int nsort = in.shape(axis);
  int nsegments = in.data_size() / nsort;
  int last_dim = in.ndim() - 1;

  // If we are not sorting the innermost dimension of a contiguous array,
  // transpose and make a copy.
  bool is_segmented_sort = in.flags().contiguous && in.strides()[axis] == 1;
  if (!is_segmented_sort) {
    array trans = swapaxes_in_eval(in, axis, last_dim);
    in = array(trans.shape(), trans.dtype(), nullptr, {});
    copy_gpu(trans, in, CopyType::General, s);
    encoder.add_temporary(in);
    out = array(allocator::malloc(out.nbytes()), in.shape(), out.dtype());
    encoder.add_temporary(out);
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }

  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE, {
      if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
        using Type = cuda_type_t<CTYPE>;
        auto offsets = thrust::make_transform_iterator(
            thrust::make_counting_iterator(0),
            [nsort] __device__(int i) { return i * nsort; });
        if (argsort) {
          // Indices in the sorted dimension.
          array indices(
              allocator::malloc(out.nbytes()), in.shape(), out.dtype());
          encoder.add_temporary(indices);
          thrust::transform(
              cu::thrust_policy(stream),
              thrust::counting_iterator<uint32_t>(0),
              thrust::counting_iterator<uint32_t>(indices.data_size()),
              thrust::device_pointer_cast(indices.data<uint32_t>()),
              ModOp<uint32_t>{static_cast<uint32_t>(nsort)});

          // In argsort though we don't need the result of sorted values, the
          // API requires us to provide an array to store it.
          array discard(allocator::malloc(in.nbytes()), in.shape(), in.dtype());
          encoder.add_temporary(discard);

          segmented_sort_pairs(
              encoder,
              in.data<Type>(),
              discard.data<Type>(),
              indices.data<uint32_t>(),
              out.data<uint32_t>(),
              in.data_size(),
              nsegments,
              offsets,
              offsets + 1,
              stream);
        } else {
          segmented_sort(
              encoder,
              in.data<Type>(),
              out.data<Type>(),
              in.data_size(),
              nsegments,
              offsets,
              offsets + 1,
              stream);
        }
      } else {
        throw std::runtime_error(
            "CUDA backend does not support sorting complex numbers");
      }
    });
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

} // namespace mlx::core
