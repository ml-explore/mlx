// Copyright © 2023 Apple Inc.

#pragma once

#include <vector>

#include "mlx/array.h"

namespace mlx::core {

inline size_t elem_to_loc(
    int elem,
    const std::vector<int>& shape,
    const std::vector<size_t>& strides) {
  size_t loc = 0;
  for (int i = shape.size() - 1; i >= 0; --i) {
    auto q_and_r = ldiv(elem, shape[i]);
    loc += q_and_r.rem * strides[i];
    elem = q_and_r.quot;
  }
  return loc;
}

inline size_t elem_to_loc(int elem, const array& a) {
  if (a.flags().row_contiguous) {
    return elem;
  }
  return elem_to_loc(elem, a.shape(), a.strides());
}

} // namespace mlx::core
