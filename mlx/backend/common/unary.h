// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/common/utils.h"

namespace mlx::core {

inline void set_unary_output_data(
    const array& in,
    array& out,
    std::function<allocator::Buffer(size_t)> mallocfn = allocator::malloc) {
  if (in.flags().contiguous) {
    if (is_donatable(in, out)) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          mallocfn(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    out.set_data(mallocfn(out.nbytes()));
  }
}

} // namespace mlx::core
