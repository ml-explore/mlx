// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/iterators/cast_iterator.cuh"
#include "mlx/backend/cuda/kernels/iterators/general_iterator.cuh"
#include "mlx/backend/cuda/kernels/iterators/repeat_iterator.cuh"
#include "mlx/backend/cuda/kernels/utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/primitives.h"

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <cassert>

namespace mlx::core {

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int64_t inp_offset,
    int64_t out_offset,
    CopyType ctype,
    const Stream& s,
    const std::optional<array>& dynamic_i_offset /* = std::nullopt */,
    const std::optional<array>& dynamic_o_offset /* = std::nullopt */) {
  if (out.size() == 0) {
    return;
  }
  // TODO: Figure out how to handle donated input.
  assert(in.data_shared_ptr() != nullptr);

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE_IN, {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, {
        using InType = cuda_type_t<CTYPE_IN>;
        using OutType = cuda_type_t<CTYPE_OUT>;
        if constexpr (cuda::std::is_convertible_v<InType, OutType>) {
          auto policy = cu::thrust_policy(stream);
          auto in_ptr = cu::make_cast_iterator<OutType>(
              thrust::device_pointer_cast(in.data<InType>() + inp_offset));
          auto out_ptr =
              thrust::device_pointer_cast(out.data<OutType>() + out_offset);
          if (ctype == CopyType::Scalar) {
            thrust::copy_n(
                policy, cu::repeat_iterator(in_ptr), out.data_size(), out_ptr);
          } else if (ctype == CopyType::Vector) {
            thrust::copy_n(policy, in_ptr, out.data_size(), out_ptr);
          } else {
            bool dynamic = dynamic_i_offset || dynamic_o_offset;
            if (dynamic) {
              throw std::runtime_error(
                  "Dynamic copy not implemented for CUDA backend.");
            }
            auto [shape_collapsed, strides_vec] = collapse_contiguous_dims(
                shape,
                std::vector{strides_in, strides_out},
                /* size_cap = */ INT32_MAX);
            if (ctype == CopyType::General) {
              thrust::copy_n(
                  policy,
                  cu::make_general_iterator<int64_t>(
                      in_ptr, shape_collapsed, strides_vec[0]),
                  out.data_size(),
                  out_ptr);
            } else {
              thrust::copy_n(
                  policy,
                  cu::make_general_iterator<int64_t>(
                      in_ptr, shape_collapsed, strides_vec[0]),
                  out.data_size(),
                  cu::make_general_iterator<int64_t>(
                      out_ptr, shape_collapsed, strides_vec[1]));
            }
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not copy data from dtype {} to {}.",
              dtype_to_string(in.dtype()),
              dtype_to_string(out.dtype())));
        }
      });
    });
  });
}

void fill_gpu(const array& val, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(val);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_ALL_TYPES(val.dtype(), CTYPE_IN, {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, {
        using InType = cuda_type_t<CTYPE_IN>;
        using OutType = cuda_type_t<CTYPE_OUT>;
        if constexpr (cuda::std::is_convertible_v<InType, OutType>) {
          thrust::copy_n(
              cu::thrust_policy(stream),
              cu::make_cast_iterator<OutType>(cu::repeat_iterator(
                  thrust::device_pointer_cast(val.data<InType>()))),
              out.data_size(),
              thrust::device_pointer_cast(out.data<OutType>()));
        } else {
          throw std::runtime_error(fmt::format(
              "Can not fill data of dtype {} with {}",
              dtype_to_string(out.dtype()),
              dtype_to_string(val.dtype())));
        }
      });
    });
  });
}

} // namespace mlx::core
