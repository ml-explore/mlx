// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/utils.h"

namespace mlx::core {

void transpose(const array& in, array& out, const std::vector<int>& axes) {
  Strides out_strides(out.ndim());
  for (int ax = 0; ax < axes.size(); ++ax) {
    out_strides[ax] = in.strides()[axes[ax]];
  }

  // Conditions for {row/col}_contiguous
  // - array must be contiguous (no gaps)
  // - underlying buffer size should have the same size as the array
  // - cumulative product of shapes is equal to the strides (we can ignore axes
  //   with size == 1)
  //   - in the forward direction (column contiguous)
  //   - in the reverse direction (row contiguous)
  // - vectors are both row and col contiguous (hence if both row/col are
  //   true, they stay true)
  auto flags = in.flags();
  if (flags.contiguous && in.data_size() == in.size()) {
    auto [_, rc, cc] = check_contiguity(out.shape(), out_strides);
    flags.row_contiguous = rc;
    flags.col_contiguous = cc;
  }
  out.copy_shared_buffer(in, out_strides, flags, in.data_size());
}

} // namespace mlx::core
