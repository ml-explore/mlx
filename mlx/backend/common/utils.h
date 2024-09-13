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

template <typename stride_t>
std::vector<stride_t> make_contiguous_strides(const std::vector<int>& shape) {
  std::vector<stride_t> strides(shape.size(), 1);
  for (int i = shape.size() - 1; i > 0; i--) {
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Collapse dims that are contiguous to possibly route to a better kernel
// e.g. for x = transpose(array({0, 1, 2, 3, 4, 5, 6, 7}, {2, 2, 2}), {2, 0, 1})
// should return {{2, 4}, {{1, 2}}}.
//
// When multiple arrays are passed they should all have the same shape. The
// collapsed axes are also the same so one shape is returned.
std::tuple<std::vector<int>, std::vector<std::vector<int64_t>>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<std::vector<int64_t>>& strides,
    int64_t size_cap = std::numeric_limits<int64_t>::max());
std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<std::vector<size_t>>& strides,
    size_t size_cap = std::numeric_limits<size_t>::max());

inline std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(
    const std::vector<array>& xs,
    size_t size_cap = std::numeric_limits<size_t>::max()) {
  std::vector<std::vector<size_t>> strides;
  for (auto& x : xs) {
    strides.emplace_back(x.strides());
  }
  return collapse_contiguous_dims(xs[0].shape(), strides, size_cap);
}

template <typename... Arrays, typename = enable_for_arrays_t<Arrays...>>
inline auto collapse_contiguous_dims(Arrays&&... xs) {
  return collapse_contiguous_dims(
      std::vector<array>{std::forward<Arrays>(xs)...});
}

// The single array version of the above.
inline std::tuple<std::vector<int>, std::vector<size_t>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<size_t>& strides) {
  std::vector<int> collapsed_shape;
  std::vector<size_t> collapsed_strides;

  if (shape.size() > 0) {
    collapsed_shape.push_back(shape[0]);
    collapsed_strides.push_back(strides[0]);
    for (int i = 1; i < shape.size(); i++) {
      if (strides[i] * shape[i] != collapsed_strides.back() ||
          collapsed_shape.back() * static_cast<size_t>(shape[i]) >
              std::numeric_limits<int>::max()) {
        collapsed_shape.push_back(shape[i]);
        collapsed_strides.push_back(strides[i]);
      } else {
        collapsed_shape.back() *= shape[i];
        collapsed_strides.back() = strides[i];
      }
    }
  }

  return std::make_tuple(collapsed_shape, collapsed_strides);
}

template <typename stride_t>
inline auto check_contiguity(
    const std::vector<int>& shape,
    const std::vector<stride_t>& strides) {
  size_t no_broadcast_data_size = 1;
  size_t f_stride = 1;
  size_t b_stride = 1;
  bool is_row_contiguous = true;
  bool is_col_contiguous = true;

  for (int i = 0, ri = shape.size() - 1; ri >= 0; i++, ri--) {
    is_col_contiguous &= strides[i] == f_stride || shape[i] == 1;
    is_row_contiguous &= strides[ri] == b_stride || shape[ri] == 1;
    f_stride *= shape[i];
    b_stride *= shape[ri];
    if (strides[i] > 0) {
      no_broadcast_data_size *= shape[i];
    }
  }

  return std::make_tuple(
      no_broadcast_data_size, is_row_contiguous, is_col_contiguous);
}

inline bool is_donatable(const array& in, const array& out) {
  constexpr size_t donation_extra = 16384;

  return in.is_donatable() && in.itemsize() == out.itemsize() &&
      in.buffer_size() <= out.nbytes() + donation_extra;
}

} // namespace mlx::core
