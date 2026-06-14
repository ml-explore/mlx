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

std::pair<dim3, dim3> get_grid_and_block(int dim0, int dim1, int dim2) {
  auto [grid, block] = get_grid_and_block_common(dim0, dim1, dim2);
  auto [gx, gy, gz] = grid;
  auto [bx, by, bz] = block;
  return std::make_pair(dim3(gx, gy, gz), dim3(bx, by, bz));
}

std::tuple<dim3, uint32_t> get_launch_args(
    size_t size,
    const Shape& shape,
    const Strides& strides,
    bool large,
    int work_per_thread /* = 1 */,
    uint32_t max_block_dim /* = 1024 */) {
  size_t nthreads = cuda::ceil_div(size, work_per_thread);
  uint32_t block_dim = max_block_dim < nthreads ? max_block_dim : nthreads;
  dim3 num_blocks;
  if (large) {
    num_blocks = get_2d_grid_dims(shape, strides, work_per_thread);
    num_blocks.x = cuda::ceil_div(num_blocks.x, block_dim);
  } else {
    num_blocks.x = cuda::ceil_div(nthreads, block_dim);
  }
  return std::make_tuple(num_blocks, block_dim);
}

std::pair<dim3, dim3>
get_launch_args_general(int dim0, size_t rest, int work_per_thread /* = 1 */) {
  constexpr uint32_t max_grid_yz_dim = 65535;
  dim0 = (dim0 + work_per_thread - 1) / work_per_thread;
  dim3 block_dims = get_block_dims(dim0, rest, 1);
  uint32_t num_blocks_x = cuda::ceil_div(dim0, block_dims.x);
  uint32_t num_blocks_y =
      static_cast<uint32_t>(cuda::ceil_div(rest, size_t{block_dims.y}));

  uint32_t num_blocks_z = cuda::ceil_div(num_blocks_y, max_grid_yz_dim);
  num_blocks_y = cuda::ceil_div(num_blocks_y, num_blocks_z);
  return {dim3(num_blocks_x, num_blocks_y, num_blocks_z), block_dims};
}

} // namespace mlx::core
