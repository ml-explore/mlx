// Copyright © 2023-2024 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/common/utils.h"

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

void copy(const array& src, array& dst, CopyType ctype);
void copy_inplace(const array& src, array& dst, CopyType ctype);

void copy_inplace(
    const array& src,
    array& dst,
    const Shape& data_shape,
    const Strides& i_strides,
    const Strides& o_strides,
    int64_t i_offset,
    int64_t o_offset,
    CopyType ctype);

} // namespace mlx::core
