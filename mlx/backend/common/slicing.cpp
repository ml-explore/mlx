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
  auto [_, is_row_contiguous, is_col_contiguous] =
      check_contiguity(out.shape(), out_strides);

  auto flags = in.flags();
  flags.row_contiguous = is_row_contiguous;
  flags.col_contiguous = is_col_contiguous;

  if (data_size == 1) {
    // Broadcasted scalar array is contiguous.
    flags.contiguous = true;
  } else if (data_size == in.data_size()) {
    // Means we sliced a broadcasted dimension so leave the "no holes" flag
    // alone.
  } else {
    // We sliced something. So either we are row or col contiguous or we
    // punched a hole.
    flags.contiguous &= flags.row_contiguous || flags.col_contiguous;
  }

  out.copy_shared_buffer(in, out_strides, flags, data_size, data_offset);
}

} // namespace mlx::core
