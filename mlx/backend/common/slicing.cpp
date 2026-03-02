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
    int64_t data_offset,
    size_t data_size,
    array& out) {
  // Compute row/col contiguity
  auto [no_bsx_size, is_row_contiguous, is_col_contiguous] =
      check_contiguity(out.shape(), out_strides);

  auto flags = in.flags();
  flags.row_contiguous = is_row_contiguous;
  flags.col_contiguous = is_col_contiguous;
  flags.contiguous = (no_bsx_size == data_size);

  out.copy_shared_buffer(in, out_strides, flags, data_size, data_offset);
}

void slice(
    const array& in,
    array& out,
    const Shape& start_indices,
    const Shape& strides) {
  if (out.size() == 0) {
    out.set_data(allocator::malloc(0));
    return;
  }

  // Calculate out strides, initial offset
  auto [data_offset, inp_strides] = prepare_slice(in, start_indices, strides);

  // Get the location of the end based on the inp strides and out.shape()
  int64_t low_idx = 0;
  int64_t high_idx = 0;
  for (int i = 0; i < inp_strides.size(); ++i) {
    auto delta = inp_strides[i] * (out.shape()[i] - 1);
    if (inp_strides[i] > 0) {
      high_idx += delta;
    } else {
      low_idx += delta;
    }
  }
  int64_t data_size = (high_idx - low_idx) + 1;
  if (data_size < 0) {
    std::ostringstream msg;
    msg << "[slice] Computed invalid data size: " << data_size << ".";
    throw std::runtime_error(msg.str());
  }
  shared_buffer_slice(in, inp_strides, data_offset, data_size, out);
}

} // namespace mlx::core
