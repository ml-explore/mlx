// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::string get_primitive_string(Primitive* primitive) {
  std::ostringstream op_t;
  primitive->print(op_t);
  return op_t.str();
}

std::tuple<Shape, std::vector<Strides>> collapse_contiguous_dims(
    const Shape& shape,
    const std::vector<Strides>& strides,
    int64_t size_cap) {
  // Make a vector that has axes separated with -1. Collapse all axes between
  // -1.
  Shape to_collapse;
  if (shape.size() > 0) {
    if (shape[0] != 1) {
      to_collapse.push_back(0);
    }
    size_t size = shape[0];
    for (int i = 1; i < shape.size(); i++) {
      bool contiguous = true;
      size *= shape[i];
      for (const auto& st : strides) {
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

  Shape out_shape;
  std::vector<Strides> out_strides(strides.size());
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
      const auto& st = strides[j];
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

std::pair<Shape, Strides> collapse_contiguous_dims(
    const Shape& shape,
    const Strides& strides,
    int64_t size_cap) {
  Shape collapsed_shape;
  Strides collapsed_strides;

  if (shape.size() > 0) {
    collapsed_shape.push_back(shape[0]);
    collapsed_strides.push_back(strides[0]);
    for (int i = 1; i < shape.size(); i++) {
      if (shape[i] == 1) {
        continue;
      } else if (
          strides[i] * shape[i] != collapsed_strides.back() ||
          collapsed_shape.back() * static_cast<int64_t>(shape[i]) > size_cap) {
        collapsed_shape.push_back(shape[i]);
        collapsed_strides.push_back(strides[i]);
      } else {
        collapsed_shape.back() *= shape[i];
        collapsed_strides.back() = strides[i];
      }
    }
  }

  return std::make_pair(collapsed_shape, collapsed_strides);
}

std::pair<Shape, Strides> collapse_contiguous_dims(
    const array& a,
    int64_t size_cap /* = std::numeric_limits<int32_t>::max()*/) {
  return collapse_contiguous_dims(a.shape(), a.strides(), size_cap);
}

} // namespace mlx::core
