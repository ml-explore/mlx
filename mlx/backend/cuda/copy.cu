// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/iterators/general_iterator.cuh"
#include "mlx/backend/cuda/kernels/iterators/repeat_iterator.cuh"
#include "mlx/backend/cuda/kernels/utils.cuh"
#include "mlx/backend/metal/copy.h"
#include "mlx/primitives.h"

#include <assert.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>

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

  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  encoder.launch_thrust([&](auto policy) {
    MLX_SWITCH_ALL_TYPES(in.dtype(), CTYPE_IN, [&]() {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, [&]() {
        using InType = cuda_type_t<CTYPE_IN>;
        using OutType = cuda_type_t<CTYPE_OUT>;
        if constexpr (std::is_convertible_v<InType, OutType>) {
          auto in_ptr =
              thrust::device_pointer_cast(in.data<InType>() + inp_offset);
          auto out_ptr =
              thrust::device_pointer_cast(out.data<OutType>() + out_offset);
          if (ctype == CopyType::Scalar) {
            thrust::copy_n(
                policy,
                mxcuda::make_repeat_iterator(in_ptr),
                out.data_size(),
                out_ptr);
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
            auto& strides_in_collapsed = strides_vec[0];
            auto& strides_out_collapsed = strides_vec[1];
            if (ctype == CopyType::General) {
              thrust::copy_n(
                  policy,
                  mxcuda::make_general_iterator<int64_t>(
                      in_ptr, shape_collapsed, strides_in_collapsed),
                  out.data_size(),
                  out_ptr);
            } else {
              thrust::copy_n(
                  policy,
                  mxcuda::make_general_iterator<int64_t>(
                      in_ptr, shape_collapsed, strides_in_collapsed),
                  out.data_size(),
                  mxcuda::make_general_iterator<int64_t>(
                      out_ptr, shape_collapsed, strides_in_collapsed));
            }
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not copy data from dtype {} to {}",
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
  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(val);
  encoder.set_output_array(out);
  encoder.launch_thrust([&](auto policy) {
    MLX_SWITCH_ALL_TYPES(val.dtype(), CTYPE_IN, [&]() {
      MLX_SWITCH_ALL_TYPES(out.dtype(), CTYPE_OUT, [&]() {
        using InType = cuda_type_t<CTYPE_IN>;
        using OutType = cuda_type_t<CTYPE_OUT>;
        if constexpr (std::is_convertible_v<InType, OutType>) {
          thrust::copy_n(
              policy,
              mxcuda::make_repeat_iterator(
                  thrust::device_pointer_cast(val.data<InType>())),
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

// TODO: Code below are identical to backend/metal/copy.cpp
void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  bool donated = set_copy_output_data(in, out, ctype);
  if (donated && in.dtype() == out.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    return;
  }
  if (ctype == CopyType::GeneralGeneral) {
    ctype = CopyType::General;
  }
  copy_gpu_inplace(in, out, ctype, s);
}

void copy_gpu(const array& in, array& out, CopyType ctype) {
  copy_gpu(in, out, ctype, out.primitive().stream());
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    CopyType ctype,
    const Stream& s) {
  assert(in.shape() == out.shape());
  return copy_gpu_inplace(
      in, out, in.shape(), in.strides(), out.strides(), 0, 0, ctype, s);
}

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Strides& i_strides,
    int64_t i_offset,
    CopyType ctype,
    const Stream& s) {
  assert(in.shape() == out.shape());
  return copy_gpu_inplace(
      in, out, in.shape(), i_strides, out.strides(), i_offset, 0, ctype, s);
}

} // namespace mlx::core
