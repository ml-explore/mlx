// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/copy/copy.cuh"

namespace mlx::core {

void copy_gpu_inplace(
    const array& in,
    array& out,
    const Shape& shape,
    const Strides& strides_in,
    const Strides& strides_out,
    int64_t offset_in,
    int64_t offset_out,
    CopyType ctype,
    const Stream& s,
    const std::optional<array>& dynamic_offset_in,
    const std::optional<array>& dynamic_offset_out) {
  if (out.size() == 0) {
    return;
  }

  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  if (ctype == CopyType::Scalar || ctype == CopyType::Vector) {
    copy_contiguous(encoder, ctype, in, out, offset_in, offset_out);
    return;
  }

  if (ctype == CopyType::General || ctype == CopyType::GeneralGeneral) {
    auto [shape_collapsed, strides_vec] = collapse_contiguous_dims(
        shape, std::vector{strides_in, strides_out}, INT32_MAX);
    if (ctype == CopyType::General) {
      copy_general_input(
          encoder,
          ctype,
          in,
          out,
          offset_in,
          offset_out,
          shape_collapsed,
          strides_vec[0]);
    } else {
      if (dynamic_offset_in || dynamic_offset_out) {
        copy_general_dynamic(
            encoder,
            ctype,
            in,
            out,
            offset_in,
            offset_out,
            shape_collapsed,
            strides_vec[0],
            strides_vec[1],
            dynamic_offset_in ? *dynamic_offset_in : array(0, int64),
            dynamic_offset_out ? *dynamic_offset_out : array(0, int64));
      } else {
        copy_general(
            encoder,
            ctype,
            in,
            out,
            offset_in,
            offset_out,
            shape_collapsed,
            strides_vec[0],
            strides_vec[1]);
      }
    }
    return;
  }
}

void fill_gpu(const array& in, array& out, const Stream& s) {
  if (out.size() == 0) {
    return;
  }
  out.set_data(allocator::malloc(out.nbytes()));
  auto& encoder = cu::get_command_encoder(s);
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  copy_contiguous(encoder, CopyType::Scalar, in, out, 0, 0);
}

} // namespace mlx::core
