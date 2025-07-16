// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/utils.h"
#include "mlx/backend/common/utils.h"

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

MTL::Size get_block_dims(int dim0, int dim1, int dim2, int pow2) {
  Dims dims = get_block_dims_common(dim0, dim1, dim2, pow2);
  return MTL::Size(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}

MTL::Size get_2d_grid_dims(const Shape& shape, const Strides& strides) {
  Dims dims = get_2d_grid_dims_common(shape, strides);
  return MTL::Size(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}

MTL::Size
get_2d_grid_dims(const Shape& shape, const Strides& strides, size_t divisor) {
  Dims dims = get_2d_grid_dims_common(shape, strides, divisor);
  return MTL::Size(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}

} // namespace mlx::core
