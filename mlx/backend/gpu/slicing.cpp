// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/gpu/slicing.h"

namespace mlx::core {

void slice_gpu(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides,
    const Stream& s) {
  slice(in, out, start_indices, strides);
}

void pad_gpu(
    const array& in,
    const array& val,
    array& out,
    const std::vector<int>& axes,
    const Shape& low_pad_size,
    const Stream& s) {
  // Fill output with val
  fill_gpu(val, out, s);

  // Find offset for start of input values
  size_t data_offset = 0;
  for (int i = 0; i < axes.size(); i++) {
    auto ax = axes[i] < 0 ? out.ndim() + axes[i] : axes[i];
    data_offset += out.strides()[ax] * low_pad_size[i];
  }

  // Extract slice from output where input will be pasted
  array out_slice(in.shape(), out.dtype(), nullptr, {});
  out_slice.copy_shared_buffer(
      out, out.strides(), out.flags(), out_slice.size(), data_offset);

  // Copy input values into the slice
  copy_gpu_inplace(in, out_slice, CopyType::GeneralGeneral, s);
}

} // namespace mlx::core
