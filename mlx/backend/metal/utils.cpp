// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/utils.h"

namespace mlx::core {

std::string type_to_name(const Dtype& t) {
  std::string tname;
  switch (t) {
    case bool_:
      tname = "bool_";
      break;
    case uint8:
      tname = "uint8";
      break;
    case uint16:
      tname = "uint16";
      break;
    case uint32:
      tname = "uint32";
      break;
    case uint64:
      tname = "uint64";
      break;
    case int8:
      tname = "int8";
      break;
    case int16:
      tname = "int16";
      break;
    case int32:
      tname = "int32";
      break;
    case int64:
      tname = "int64";
      break;
    case float16:
      tname = "float16";
      break;
    case float32:
      tname = "float32";
      break;
    case float64:
      tname = "double";
      break;
    case bfloat16:
      tname = "bfloat16";
      break;
    case complex64:
      tname = "complex64";
      break;
  }
  return tname;
}

std::string type_to_name(const array& a) {
  return type_to_name(a.dtype());
}

MTL::Size get_block_dims(int dim0, int dim1, int dim2, int pow2 /* = 10 */) {
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
  return MTL::Size{1ul << pows[0], 1ul << pows[1], 1ul << pows[2]};
}

MTL::Size get_2d_grid_dims(const Shape& shape, const Strides& strides) {
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
  return MTL::Size(
      static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1);
}

MTL::Size
get_2d_grid_dims(const Shape& shape, const Strides& strides, size_t divisor) {
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
  if (grid_y > UINT32_MAX || grid_x > UINT32_MAX || divisor > 1) {
    throw std::runtime_error("Unable to safely factor shape.");
  }
  if (grid_y > grid_x) {
    std::swap(grid_x, grid_y);
  }
  return MTL::Size(
      static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1);
}

} // namespace mlx::core
