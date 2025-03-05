// Copyright © 2023-2024 Apple Inc.

#include <sstream>

#include "mlx/backend/metal/copy.h"
#include "mlx/backend/webgpu/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

constexpr int MAX_COPY_SPECIALIZED_DIMS = 3;

void copy_gpu(const array& in, array& out, CopyType ctype, const Stream& s) {
  if (ctype == CopyType::Vector) {
    // If the input is donateable, we are doing a vector copy and the types
    // have the same size, then the input buffer can hold the output.
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.move_shared_buffer(in);
      // If the output has the same type as the input then there is nothing to
      // copy, just use the buffer.
      if (in.dtype() == out.dtype()) {
        return;
      }
    } else {
      out.set_data(
          webgpu::allocator().malloc_gpu(out, in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    out.set_data(webgpu::allocator().malloc_gpu(out, out.nbytes()));
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
  if (inp_offset > 0 || out_offset > 0) {
    throw std::runtime_error("Offset not implemented in copy.");
  }
  if (dynamic_i_offset || dynamic_o_offset) {
    throw std::runtime_error("Dynamic offset not implemented in copy.");
  }

  bool donate_in = in.data_shared_ptr() == nullptr;
  const betann::Buffer& src = get_gpu_buffer(donate_in ? out : in);

  auto& device = webgpu::device(s.device);
  if (ctype == CopyType::General) {
    betann::CopyGeneral(
        device,
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        dtype_to_webgpu(in.dtype()),
        src,
        to_u32_vector(shape),
        to_u32_vector(strides_in));
  } else if (ctype == CopyType::GeneralGeneral) {
    betann::CopyGeneralBoth(
        device,
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        to_u32_vector(strides_out),
        dtype_to_webgpu(in.dtype()),
        src,
        to_u32_vector(shape),
        to_u32_vector(strides_in));
  } else {
    betann::CopyContiguous(
        device,
        static_cast<betann::CopyType>(ctype),
        dtype_to_webgpu(out.dtype()),
        get_gpu_buffer(out),
        out.data_size(),
        dtype_to_webgpu(in.dtype()),
        src);
  }
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
