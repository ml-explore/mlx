// Copyright © 2024 Apple Inc.

#include <numeric>

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core {

void slice_gpu(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides,
    const Stream& s) {
  // Calculate out strides and initial offset
  auto [data_offset, inp_strides] = prepare_slice(in, start_indices, strides);

  size_t data_end = 1;
  for (int i = 0; i < strides.size(); ++i) {
    if (in.shape()[i] > 1) {
      auto end_idx = start_indices[i] + out.shape()[i] * strides[i] - 1;
      data_end += end_idx * in.strides()[i];
    }
  }
  size_t data_size = data_end - data_offset;
  shared_buffer_slice(in, inp_strides, data_offset, data_size, out);
}

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& out,
    int axis,
    const Stream& s) {
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  out.set_data(allocator::malloc_or_wait(out.nbytes()));

  auto strides = out.strides();
  auto flags = out.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;
  auto& d = metal::device(s.device);
  auto& compute_encoder = d.get_command_encoder(s.index);
  auto concurrent_ctx = compute_encoder.start_concurrent();
  for (int i = 0; i < inputs.size(); i++) {
    array out_slice(inputs[i].shape(), out.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        out, strides, flags, out_slice.size(), data_offset);
    copy_gpu_inplace(inputs[i], out_slice, CopyType::GeneralGeneral, s);
  }
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
