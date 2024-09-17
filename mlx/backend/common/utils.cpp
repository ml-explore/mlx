// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/utils.h"

namespace mlx::core {

template <typename stride_t>
std::tuple<std::vector<int>, std::vector<std::vector<stride_t>>>
collapse_contiguous_dims_impl(
    const std::vector<int>& shape,
    const std::vector<std::vector<stride_t>>& strides,
    stride_t size_cap) {
  // Make a vector that has axes separated with -1. Collapse all axes between
  // -1.
  std::vector<int> to_collapse;
  if (shape.size() > 0) {
    if (shape[0] != 1) {
      to_collapse.push_back(0);
    }
    size_t size = shape[0];
    for (int i = 1; i < shape.size(); i++) {
      bool contiguous = true;
      size *= shape[i];
      for (const std::vector<stride_t>& st : strides) {
        if (st[i] * shape[i] != st[i - 1] || size > size_cap) {
          contiguous = false;
          size = shape[i];
          break;
        }
      }
      if (!contiguous) {
        to_collapse.push_back(-1);
      }
      if (shape[i] != 1) {
        to_collapse.push_back(i);
      }
    }
    to_collapse.push_back(-1);
  }

  std::vector<int> out_shape;
  std::vector<std::vector<stride_t>> out_strides(strides.size());
  for (int i = 0;;) {
    while (i < to_collapse.size() && to_collapse[i] == -1) {
      ++i;
    };
    if (i == to_collapse.size()) {
      break;
    }
    int current_shape = shape[to_collapse[i]];
    int k = i;
    while (to_collapse[++k] != -1) {
      current_shape *= shape[to_collapse[k]];
    }
    out_shape.push_back(current_shape);
    for (int j = 0; j < strides.size(); j++) {
      const std::vector<stride_t>& st = strides[j];
      out_strides[j].push_back(st[to_collapse[k - 1]]);
    }
    i = k + 1;
  }

  if (!shape.empty() && out_shape.empty()) {
    out_shape.push_back(1);
    for (auto& out_stride : out_strides) {
      out_stride.push_back(0);
    }
  }
  return std::make_tuple(out_shape, out_strides);
}

std::tuple<std::vector<int>, std::vector<std::vector<int64_t>>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<std::vector<int64_t>>& strides,
    int64_t size_cap /* = std::numeric_limits<int32_t>::max() */) {
  return collapse_contiguous_dims_impl(shape, strides, size_cap);
}

std::tuple<std::vector<int>, std::vector<std::vector<size_t>>>
collapse_contiguous_dims(
    const std::vector<int>& shape,
    const std::vector<std::vector<size_t>>& strides,
    size_t size_cap /* = std::numeric_limits<int32>::max() */) {
  return collapse_contiguous_dims_impl(shape, strides, size_cap);
}

} // namespace mlx::core
