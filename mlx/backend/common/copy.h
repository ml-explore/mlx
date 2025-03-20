// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/array.h"

namespace mlx::core {

enum class CopyType {
  // Copy a raw scalar input into the full contiguous output
  Scalar,

  // Copy the raw input buffer contiguously into a raw output buffer of the same
  // size
  Vector,

  // Copy the full virtual input to the full contiguous output
  General,

  // Copy the full virtual input to the full virtual output. We assume the
  // input and output have the same shape.
  GeneralGeneral
};

inline bool set_copy_output_data(const array& in, array& out, CopyType ctype) {
  if (ctype == CopyType::Vector) {
    // If the input is donateable, we are doing a vector copy and the types
    // have the same size, then the input buffer can hold the output.
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
      return true;
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
      return false;
    }
  } else {
    out.set_data(allocator::malloc(out.nbytes()));
    return false;
  }
}

} // namespace mlx::core
