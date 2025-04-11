// Copyright © 2024 Apple Inc.

#include "mlx/backend/common/utils.h"

namespace mlx::core {

void broadcast(const array& in, array& out) {
  if (out.size() == 0) {
    out.set_data(nullptr);
    return;
  }
  Strides strides(out.ndim(), 0);
  int diff = out.ndim() - in.ndim();
  for (int i = in.ndim() - 1; i >= 0; --i) {
    strides[i + diff] = (in.shape()[i] == 1) ? 0 : in.strides()[i];
  }
  auto flags = in.flags();
  if (out.size() > in.size()) {
    flags.row_contiguous = flags.col_contiguous = false;
  }
  out.copy_shared_buffer(in, strides, flags, in.data_size());
}

} // namespace mlx::core
