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
    const Stream&) {
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

  // Find start of input values
  Shape start(out.ndim(), 0);
  for (int i = 0; i < axes.size(); i++) {
    auto ax = axes[i] < 0 ? out.ndim() + axes[i] : axes[i];
    start[ax] += low_pad_size[i];
  }

  // Extract slice from output where input will be pasted
  array out_slice(in.shape(), out.dtype(), nullptr, {});
  slice(out, out_slice, start, Shape(out.ndim(), 1));

  // Copy input values into the slice
  copy_gpu_inplace(in, out_slice, CopyType::GeneralGeneral, s);
}

} // namespace mlx::core
