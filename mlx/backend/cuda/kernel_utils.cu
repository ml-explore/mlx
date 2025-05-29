// Copyright © 2025 Apple Inc.

#include "mlx/backend/common/utils.h"
#include "mlx/backend/cuda/kernel_utils.cuh"

namespace mlx::core {

dim3 get_block_dims(int dim0, int dim1, int dim2, int pow2) {
  Dims dims = get_block_dims_common(dim0, dim1, dim2, pow2);
  return dim3(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}

dim3 get_2d_grid_dims(const Shape& shape, const Strides& strides) {
  Dims dims = get_2d_grid_dims_common(shape, strides);
  return dim3(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}

dim3 get_2d_grid_dims(
    const Shape& shape,
    const Strides& strides,
    size_t divisor) {
  Dims dims = get_2d_grid_dims_common(shape, strides, divisor);
  return dim3(std::get<0>(dims), std::get<1>(dims), std::get<2>(dims));
}

} // namespace mlx::core
