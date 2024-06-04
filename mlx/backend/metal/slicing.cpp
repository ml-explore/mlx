// Copyright © 2024 Apple Inc.

#include <numeric>

#include "mlx/backend/common/slicing.h"
#include "mlx/backend/metal/copy.h"
#include "mlx/backend/metal/device.h"

namespace mlx::core {

void slice_gpu(
    const array& in,
    array& out,
    std::vector<int> start_indices,
    std::vector<int> strides,
    const Stream& s) {
  // Calculate out strides, initial offset and if copy needs to be made
  auto [copy_needed, data_offset, inp_strides] =
      prepare_slice(in, start_indices, strides);

  // Do copy if needed
  if (copy_needed) {
    out.set_data(allocator::malloc_or_wait(out.nbytes()));
    std::vector<int64_t> ostrides{out.strides().begin(), out.strides().end()};
    copy_gpu_inplace(
        /* const array& in = */ in,
        /* array& out = */ out,
        /* const std::vector<int>& data_shape = */ out.shape(),
        /* const std::vector<stride_t>& i_strides = */ inp_strides,
        /* const std::vector<stride_t>& o_strides = */ ostrides,
        /* int64_t i_offset = */ data_offset,
        /* int64_t o_offset = */ 0,
        /* CopyType ctype = */ CopyType::General,
        /* const Stream& s = */ s);
  } else {
    std::vector<size_t> ostrides{inp_strides.begin(), inp_strides.end()};
    shared_buffer_slice(in, ostrides, data_offset, out);
  }
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
    std::vector<int> axes,
    std::vector<int> low_pad_size,
    const Stream& s) {
  // Fill output with val
  copy_gpu(val, out, CopyType::Scalar, s);

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
