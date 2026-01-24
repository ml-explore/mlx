// Copyright © 2025 Apple Inc.

#include "mlx/backend/rocm/device.h"
#include "mlx/backend/rocm/kernel_utils.hpp"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <hip/hip_runtime.h>
#include <rocprim/rocprim.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cassert>

namespace mlx::core {

namespace {

template <typename T>
struct ModOp {
  T divisor;
  __device__ T operator()(T x) const {
    return x % divisor;
  }
};

struct OffsetTransform {
  int nsort;

  __device__ int operator()(int i) const {
    return i * nsort;
  }
};

void gpu_sort(const Stream& s, array in, array& out_, int axis, bool argsort) {
  array out = out_;
  auto& encoder = rocm::get_command_encoder(s);
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
  
  auto& stream = encoder.stream();
  
  // Use rocPrim for segmented sort
  dispatch_all_types(in.dtype(), [&](auto type_tag) {
    using CTYPE = MLX_GET_TYPE(type_tag);
    if constexpr (!std::is_same_v<CTYPE, complex64_t>) {
      using Type = hip_type_t<CTYPE>;
      
      auto offsets = thrust::make_transform_iterator(
          thrust::make_counting_iterator(0), OffsetTransform{nsort});
      
      int num_segments = in.data_size() / nsort;
      
      encoder.launch_kernel([&](hipStream_t hip_stream) {
        if (argsort) {
          // Indices in the sorted dimension
          array indices(allocator::malloc(out.nbytes()), in.shape(), out.dtype());
          encoder.add_temporary(indices);

          // Discard array for sorted values (we only need indices)
          array discard(allocator::malloc(in.nbytes()), in.shape(), in.dtype());
          encoder.add_temporary(discard);

          // Initialize indices with 0, 1, 2, ... % nsort
          thrust::transform(
              thrust::hip::par.on(hip_stream),
              thrust::counting_iterator<uint32_t>(0),
              thrust::counting_iterator<uint32_t>(indices.data_size()),
              thrust::device_pointer_cast(indices.data<uint32_t>()),
              ModOp<uint32_t>{static_cast<uint32_t>(nsort)});

          // Get temp storage size
          size_t temp_size = 0;
          rocprim::segmented_radix_sort_pairs(
              nullptr,
              temp_size,
              in.data<Type>(),
              discard.data<Type>(),
              indices.data<uint32_t>(),
              out.data<uint32_t>(),
              in.data_size(),
              num_segments,
              offsets,
              offsets + 1,
              0,
              sizeof(Type) * 8,
              hip_stream);

          // Allocate temp storage
          array temp(allocator::malloc(temp_size), {static_cast<int>(temp_size)}, uint8);
          encoder.add_temporary(temp);

          // Perform sort
          rocprim::segmented_radix_sort_pairs(
              temp.data<void>(),
              temp_size,
              in.data<Type>(),
              discard.data<Type>(),
              indices.data<uint32_t>(),
              out.data<uint32_t>(),
              in.data_size(),
              num_segments,
              offsets,
              offsets + 1,
              0,
              sizeof(Type) * 8,
              hip_stream);
        } else {
          // Get temp storage size
          size_t temp_size = 0;
          rocprim::segmented_radix_sort_keys(
              nullptr,
              temp_size,
              in.data<Type>(),
              out.data<Type>(),
              in.data_size(),
              num_segments,
              offsets,
              offsets + 1,
              0,
              sizeof(Type) * 8,
              hip_stream);

          // Allocate temp storage
          array temp(allocator::malloc(temp_size), {static_cast<int>(temp_size)}, uint8);
          encoder.add_temporary(temp);

          // Perform sort
          rocprim::segmented_radix_sort_keys(
              temp.data<void>(),
              temp_size,
              in.data<Type>(),
              out.data<Type>(),
              in.data_size(),
              num_segments,
              offsets,
              offsets + 1,
              0,
              sizeof(Type) * 8,
              hip_stream);
        }
      });
    } else {
      throw std::runtime_error(
          "ROCm backend does not support sorting complex numbers");
    }
  });

  if (!is_segmented_sort) {
    // Swap the sorted axis back.
    copy_gpu(swapaxes_in_eval(out, axis, last_dim), out_, CopyType::General, s);
  }
}

} // namespace

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, true);
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  gpu_sort(stream(), inputs[0], out, axis_, false);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  gpu_sort(stream(), inputs[0], out, axis_, true);
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  gpu_sort(stream(), inputs[0], out, axis_, false);
}

} // namespace mlx::core
