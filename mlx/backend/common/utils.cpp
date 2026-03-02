// Copyright © 2023-2024 Apple Inc.

#include <dlfcn.h>

#include "mlx/backend/common/utils.h"

namespace mlx::core {

std::filesystem::path current_binary_dir() {
  static std::filesystem::path binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get current binary dir.");
    }
    return std::filesystem::path(info.dli_fname).parent_path();
  }();
  return binary_dir;
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

Dims get_block_dims_common(int dim0, int dim1, int dim2, int pow2 /* = 10 */) {
  int pows[3] = {0, 0, 0};
  int sum = 0;
  while (true) {
    int presum = sum;
    // Check all the pows
    if (dim0 >= (1 << (pows[0] + 1))) {
      pows[0]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim1 >= (1 << (pows[1] + 1))) {
      pows[1]++;
      sum++;
    }
    if (sum == 10) {
      break;
    }
    if (dim2 >= (1 << (pows[2] + 1))) {
      pows[2]++;
      sum++;
    }
    if (sum == presum || sum == pow2) {
      break;
    }
  }
  return std::make_tuple(1ul << pows[0], 1ul << pows[1], 1ul << pows[2]);
}

Dims get_2d_grid_dims_common(const Shape& shape, const Strides& strides) {
  // Dims with strides of 0 are ignored as they
  // correspond to broadcasted dimensions
  size_t grid_x = 1;
  size_t grid_y = 1;
  for (int i = 0; i < shape.size(); ++i) {
    if (strides[i] == 0) {
      continue;
    }
    if (grid_x * shape[i] < UINT32_MAX) {
      grid_x *= shape[i];
    } else {
      grid_y *= shape[i];
    }
  }
  if (grid_y > UINT32_MAX || grid_x > UINT32_MAX) {
    throw std::runtime_error("Unable to safely factor shape.");
  }
  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }
  return std::make_tuple(
      static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1);
}

Dims get_2d_grid_dims_common(
    const Shape& shape,
    const Strides& strides,
    size_t divisor) {
  // Compute the 2d grid dimensions such that the total size of the grid is
  // divided by divisor.
  size_t grid_x = 1;
  size_t grid_y = 1;
  for (int i = 0; i < shape.size(); ++i) {
    if (strides[i] == 0) {
      continue;
    }

    // No need to add this shape we can just remove it from the divisor.
    if (divisor % shape[i] == 0) {
      divisor /= shape[i];
      continue;
    }

    if (grid_x * shape[i] < UINT32_MAX) {
      grid_x *= shape[i];
    } else {
      grid_y *= shape[i];
    }

    if (divisor > 1) {
      if (grid_x % divisor == 0) {
        grid_x /= divisor;
        divisor = 1;
      } else if (grid_y % divisor == 0) {
        grid_y /= divisor;
        divisor = 1;
      }
    }
  }
  if (grid_y > UINT32_MAX || grid_x > UINT32_MAX) {
    throw std::runtime_error("Unable to safely factor shape.");
  }
  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }
  if (divisor > 1) {
    grid_x = ((grid_x + divisor - 1) / divisor) * divisor;
  }
  return std::make_tuple(
      static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1);
}

std::pair<Dims, Dims> get_grid_and_block_common(int dim0, int dim1, int dim2) {
  auto [bx, by, bz] = get_block_dims_common(dim0, dim1, dim2);
  auto gx = (dim0 + bx - 1) / bx;
  auto gy = (dim1 + by - 1) / by;
  auto gz = (dim2 + bz - 1) / bz;

  return std::make_pair(
      std::make_tuple(gx, gy, gz), std::make_tuple(bx, by, bz));
}

} // namespace mlx::core
