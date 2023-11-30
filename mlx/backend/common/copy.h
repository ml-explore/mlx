// Copyright © 2023 Apple Inc.

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

} // namespace mlx::core
