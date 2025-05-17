// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

inline void set_unary_output_data(const array& in, array& out) {
  if (in.flags().contiguous) {
    if (is_donatable(in, out)) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
  }
}

} // namespace mlx::core
