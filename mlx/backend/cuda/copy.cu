// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/dtype_utils.cuh"
#include "mlx/backend/cuda/kernels/copy.cuh"
#include "mlx/backend/metal/copy.h"
#include "mlx/primitives.h"

#include <assert.h>

namespace mlx::core {

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& data_shape,
    const Strides& strides_in_pre,
    const Strides& strides_out_pre,
    int64_t inp_offset,
    int64_t out_offset,
    CopyType ctype,
    const Stream& s,
    const std::optional<array>& dynamic_i_offset /* = std::nullopt */,
    const std::optional<array>& dynamic_o_offset /* = std::nullopt */) {
  if (out.size() == 0) {
    return;
  }
  // Try to collapse contiguous dims
  auto maybe_collapse =
      [ctype, &data_shape, &strides_in_pre, &strides_out_pre]() {
        if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
          auto [shape, strides] = collapse_contiguous_dims(
              data_shape,
              std::vector{strides_in_pre, strides_out_pre},
              /* size_cap = */ INT32_MAX);
          return std::make_tuple(shape, strides[0], strides[1]);
        } else {
          Strides e{};
          return std::make_tuple(Shape{}, e, e);
        }
      };
  auto [shape, strides_in_, strides_out_] = maybe_collapse();
  int ndim = shape.size();
  bool large;
  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    // Allow for negative strides
    large = in.data_size() > INT32_MAX || out.data_size() > INT32_MAX;
  } else {
    large = out.data_size() > UINT32_MAX;
  }
  bool dynamic = dynamic_i_offset || dynamic_o_offset;

  bool donate_in = in.data_shared_ptr() == nullptr;
  const array& input = donate_in ? out : in;

  auto& encoder = mxcuda::get_command_encoder(s);
  encoder.set_input_array(input);
  encoder.set_output_array(out);
  encoder.launch_kernel([&](cudaStream_t stream) {
    MLX_SWITCH_CUDA_TYPES(input.dtype(), CTYPE_IN, [&]() {
      MLX_SWITCH_CUDA_TYPES(out.dtype(), CTYPE_OUT, [&]() {
        if constexpr (std::is_convertible_v<CTYPE_IN, CTYPE_OUT>) {
          if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
            throw std::runtime_error(
                "General copy not implemented for CUDA backend.");
          } else {
            int num_threads = std::min(
                out.data_size(), mxcuda::max_threads_per_block(s.device));
            dim3 num_blocks = large
                ? get_2d_num_blocks(out.shape(), out.strides(), num_threads)
                : dim3(ceil_div(out.data_size(), num_threads));
            if (ctype == CopyType::Scalar) {
              mxcuda::copy_s<<<num_blocks, num_threads, 0, stream>>>(
                  input.data<CTYPE_IN>() + inp_offset,
                  out.data<CTYPE_OUT>() + out_offset,
                  out.data_size());
            } else if (ctype == CopyType::Vector) {
              mxcuda::copy_v<<<num_blocks, num_threads, 0, stream>>>(
                  input.data<CTYPE_IN>() + inp_offset,
                  out.data<CTYPE_OUT>() + out_offset,
                  out.data_size());
            }
          }
        } else {
          throw std::runtime_error(fmt::format(
              "Can not copy data from dtype {} to {}",
              dtype_to_string(input.dtype()),
              dtype_to_string(out.dtype())));
        }
      });
    });
  });
}

// TODO: Code below are identical to backend/metal/copy.cpp.
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
