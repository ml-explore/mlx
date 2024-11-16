// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/utils.h"

namespace mlx::core {

std::tuple<bool, int64_t, std::vector<int64_t>> prepare_slice(
    const array& in,
    const std::vector<int>& start_indices,
    const std::vector<int>& strides) {
  int64_t data_offset = 0;
  bool copy_needed = false;
  std::vector<int64_t> inp_strides(in.ndim(), 0);
  for (int i = 0; i < in.ndim(); ++i) {
    data_offset += start_indices[i] * in.strides()[i];
    inp_strides[i] = in.strides()[i] * strides[i];
    copy_needed |= strides[i] < 0;
  }
  return std::make_tuple(copy_needed, data_offset, inp_strides);
}

void shared_buffer_slice(
    const array& in,
    const std::vector<size_t>& out_strides,
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

} // namespace mlx::core
