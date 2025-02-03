// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/utils.h"

namespace mlx::core {

std::tuple<int64_t, Strides> prepare_slice(
    const array& in,
    const Shape& start_indices,
    const Shape& strides) {
  int64_t data_offset = 0;
  Strides inp_strides(in.ndim(), 0);
  for (int i = 0; i < in.ndim(); ++i) {
    data_offset += start_indices[i] * in.strides()[i];
    inp_strides[i] = in.strides()[i] * strides[i];
  }
  return std::make_tuple(data_offset, inp_strides);
}

void shared_buffer_slice(
    const array& in,
    const Strides& out_strides,
    size_t data_offset,
    size_t data_size,
    array& out) {
  // Compute row/col contiguity
  auto [no_bsx_size, is_row_contiguous, is_col_contiguous] =
      check_contiguity(out.shape(), out_strides);

  auto flags = in.flags();
  flags.row_contiguous = is_row_contiguous;
  flags.col_contiguous = is_col_contiguous;
  flags.contiguous = (no_bsx_size == data_size);

  move_or_copy(in, out, out_strides, flags, data_size, data_offset);
}

void slice(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides) {
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }

  // Calculate out strides, initial offset and if copy needs to be made
  auto [data_offset, inp_strides] = prepare_slice(in, start_indices, strides);
  int64_t data_end = 1;
  for (int i = 0; i < start_indices.size(); ++i) {
    if (in.shape()[i] > 1) {
      auto end_idx = start_indices[i] + out.shape()[i] * strides[i] - 1;
      data_end += end_idx * in.strides()[i];
    }
  }
  // data_end can be -1
  size_t data_size =
      data_end < 0 ? (data_offset - data_end) : (data_end - data_offset);
  shared_buffer_slice(in, inp_strides, data_offset, data_size, out);
}

} // namespace mlx::core
