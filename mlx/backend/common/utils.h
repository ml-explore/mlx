// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <vector>

#include "mlx/array.h"

namespace mlx::core {

template <typename StrideT>
inline StrideT elem_to_loc(
    int elem,
    const std::vector<int>& shape,
    const std::vector<StrideT>& strides) {
  StrideT loc = 0;
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

template <typename StrideT>
std::vector<StrideT> make_contiguous_strides(const std::vector<int>& shape) {
  std::vector<StrideT> strides(shape.size(), 1);
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
    int64_t size_cap = std::numeric_limits<int32_t>::max());
std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<std::vector<size_t>>& strides,
    size_t size_cap = std::numeric_limits<int32_t>::max());

inline std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(
    const std::vector<array>& xs,
    size_t size_cap = std::numeric_limits<int32_t>::max()) {
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
std::pair<std::vector<int>, std::vector<int64_t>> collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<int64_t>& strides,
    int64_t size_cap = std::numeric_limits<int32_t>::max());
std::pair<std::vector<int>, std::vector<size_t>> collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<size_t>& strides,
    size_t size_cap = std::numeric_limits<int32_t>::max());
std::pair<std::vector<int>, std::vector<size_t>> collapse_contiguous_dims(
    const array& a,
    size_t size_cap = std::numeric_limits<int32_t>::max());

template <typename StrideT>
struct ContiguousIterator {
  inline void step() {
    int dims = shape_.size();
    if (dims == 0) {
      return;
    }
    int i = dims - 1;
    while (pos_[i] == (shape_[i] - 1) && i > 0) {
      pos_[i] = 0;
      loc -= (shape_[i] - 1) * strides_[i];
      i--;
    }
    pos_[i]++;
    loc += strides_[i];
  }

  void seek(StrideT n) {
    loc = 0;
    for (int i = shape_.size() - 1; i >= 0; --i) {
      auto q_and_r = ldiv(n, shape_[i]);
      loc += q_and_r.rem * strides_[i];
      pos_[i] = q_and_r.rem;
      n = q_and_r.quot;
    }
  }

  void reset() {
    loc = 0;
    std::fill(pos_.begin(), pos_.end(), 0);
  }

  ContiguousIterator() {};

  explicit ContiguousIterator(const array& a)
      : shape_(a.shape()), strides_(a.strides()) {
    if (!shape_.empty()) {
      std::tie(shape_, strides_) = collapse_contiguous_dims(shape_, strides_);
      pos_ = std::vector<int>(shape_.size(), 0);
    }
  }

  explicit ContiguousIterator(
      const std::vector<int>& shape,
      const std::vector<StrideT>& strides,
      int dims)
      : shape_(shape.begin(), shape.begin() + dims),
        strides_(strides.begin(), strides.begin() + dims) {
    if (!shape_.empty()) {
      std::tie(shape_, strides_) = collapse_contiguous_dims(shape_, strides_);
      pos_ = std::vector<int>(shape_.size(), 0);
    }
  }

  StrideT loc{0};

 private:
  std::vector<int> shape_;
  std::vector<StrideT> strides_;
  std::vector<int> pos_;
};

template <typename StrideT>
inline auto check_contiguity(
    const std::vector<int>& shape,
    const std::vector<StrideT>& strides) {
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
