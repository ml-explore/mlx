// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <vector>

#include "mlx/array.h"

namespace mlx::core {

template <typename stride_t>
inline stride_t elem_to_loc(
    int elem,
    const std::vector<int>& shape,
    const std::vector<stride_t>& strides) {
  stride_t loc = 0;
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

// Collapse dims that are contiguous to possibly route to a better kernel
// e.g. for x = transpose(array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 2, 2}), {2, 0, 1})
// should return {{2, 4}, {{1, 2}}}.
//
// When multiple arrays are passed they should all have the same shape. The
// collapsed axes are also the same so one shape is returned.
template <typename stride_t>
inline std::tuple<std::vector<int>, std::vector<std::vector<stride_t>>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<std::vector<stride_t>> strides) {
  // Make a vector that has axes separated with -1. Collapse all axes between
  // -1.
  std::vector<int> to_collapse;
  if (shape.size() > 0) {
    to_collapse.push_back(0);
    for (int i = 1; i < shape.size(); i++) {
      bool contiguous = true;
      for (const std::vector<stride_t>& st : strides) {
        if (st[i] * shape[i] != st[i - 1]) {
          contiguous = false;
        }
        if (!contiguous) {
          break;
        }
      }
      if (!contiguous) {
        to_collapse.push_back(-1);
      }
      to_collapse.push_back(i);
    }
    to_collapse.push_back(-1);
  }

  std::vector<int> out_shape;
  std::vector<std::vector<stride_t>> out_strides(strides.size());
  for (int i = 0; i < to_collapse.size(); i++) {
    int current_shape = shape[to_collapse[i]];
    while (to_collapse[++i] != -1) {
      current_shape *= shape[to_collapse[i]];
    }
    out_shape.push_back(current_shape);
    for (int j = 0; j < strides.size(); j++) {
      const std::vector<stride_t>& st = strides[j];
      out_strides[j].push_back(st[to_collapse[i - 1]]);
    }
  }

  return std::make_tuple(out_shape, out_strides);
}

inline std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(const std::vector<array>& xs) {
  std::vector<std::vector<size_t>> strides;
  for (auto& x : xs) {
    strides.emplace_back(x.strides());
  }
  return collapse_contiguous_dims(xs[0].shape(), strides);
}

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
inline auto collapse_contiguous_dims(Arrays&&... xs) {
  return collapse_contiguous_dims(
      std::vector<array>{std::forward<Arrays>(xs)...});
}

template <typename stride_t>
inline auto check_contiguity(
    const std::vector<int>& shape,
    const std::vector<stride_t>& strides) {
  size_t data_size = 1;
  size_t f_stride = 1;
  size_t b_stride = 1;
  bool is_row_contiguous = true;
  bool is_col_contiguous = true;

  for (int i = 0, ri = shape.size() - 1; ri >= 0; i++, ri--) {
    is_row_contiguous &= strides[i] == f_stride || shape[i] == 1;
    is_col_contiguous &= strides[ri] == b_stride || shape[ri] == 1;
    f_stride *= shape[i];
    b_stride *= shape[ri];
    if (strides[i] > 0) {
      data_size *= shape[i];
    }
  }

  return std::make_tuple(data_size, is_row_contiguous, is_col_contiguous);
}

} // namespace mlx::core
