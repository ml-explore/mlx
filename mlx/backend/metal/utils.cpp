// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/utils.h"

using namespace mlx;

namespace mlx::core {

std::string type_to_name(const array& a) {
  std::string tname;
  switch (a.dtype()) {
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
    case bfloat16:
      tname = "bfloat16";
      break;
    case complex64:
      tname = "complex64";
      break;
  }
  return tname;
}

MTL::Size get_block_dims(int dim0, int dim1, int dim2) {
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
    if (sum == presum || sum == 10) {
      break;
    }
  }
  return MTL::Size{1ul << pows[0], 1ul << pows[1], 1ul << pows[2]};
}

MTL::Size get_2d_grid_dims(
    const std::vector<int>& shape,
    const std::vector<size_t>& strides) {
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
  return MTL::Size(
      static_cast<uint32_t>(grid_x), static_cast<uint32_t>(grid_y), 1);
}

std::string get_primitive_string(Primitive* primitive) {
  std::ostringstream op_t;
  primitive->print(op_t);
  return op_t.str();
}

} // namespace mlx::core
