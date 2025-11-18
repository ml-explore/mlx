// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cg = cooperative_groups;

// To pass scales to tensor cores, they need to be repacked into a tiled layout
// https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
// Tiled layout for scale factors is very well described in CUTLASS
// documentation:
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md#scale-factor-layouts
// Conceptually, it should be like this:
// q_w = mx.zeros(shape=(M, N)) <-- zeros just for an example
// s.shape = (M, N // 16) -- packed in row contigous order, group_size = 16
// cbg_cnt = N // 16 // 4
// rb_cnt = M // 128
// tmp = x.reshape(rb_cnt, 4, 32, cbg_cnt, 4)
// repacked_scales = tmp.transpose(0, 3, 2, 1, 4)
// example: indecis of intial tile 128 x 4 of scales (packed in row major tensor
// (M, K // 16), where M = 128, K = 64): array([[0, 1, 2, 3],
//       [4, 5, 6, 7],
//       [8, 9, 10, 11],
//       ...,
//       [500, 501, 502, 503],
//       [504, 505, 506, 507],
//       [508, 509, 510, 511]]
// packed scales within tile 128 x 4:
// array([[[[[0, 1, 2, 3], <-- s_0,0..s_0,3 scales
//          [128, 129, 130, 131], <-- s_32,0..s_32,3 scales
//          [256, 257, 258, 259], <-- s_64,0..s_64,3 scales
//          [384, 385, 386, 387]], <-- s_96,0..s_96,3 scales
//         [[4, 5, 6, 7], <-- s_1,0..s_1,3 scales
//          [132, 133, 134, 135], ...
//          [260, 261, 262, 263],
//          [388, 389, 390, 391]],
//         [[124, 125, 126, 127],
//          [252, 253, 254, 255],
//          [380, 381, 382, 383],
//          [508, 509, 510, 511]]]]],
__device__ size_t
scale_tiled_offset(size_t scale_index, size_t num_rows, size_t num_scale_cols) {
  // Compute the tiled layout offset for scale factors used in tensor cores
  // This function maps from a linear scale index to the tiled layout expected
  // by tensor cores (and cublaslt).
  //
  // Input: linear scale index (e.g., for a matrix M x K with group_size,
  //        scale_index ranges from 0 to (M * K/group_size - 1))
  //
  // The tiled layout organizes scales into tiles of 128 rows x 4 columns,
  // where each tile is subdivided into 4 sub-blocks of 32 rows x 4 columns.
  size_t row = scale_index / num_scale_cols;
  size_t col = scale_index % num_scale_cols;

  constexpr size_t rows_per_tile = 128;
  constexpr size_t rows_per_sub_block = 32;
  constexpr size_t cols_per_sub_block = 4;
  constexpr size_t sub_blocks_per_tile = 4; // Vertically stacked

  // Decompose row position
  size_t tile_row = row / rows_per_tile; // Which tile row
  size_t row_in_tile = row % rows_per_tile; // Row within tile
  size_t sub_block_row =
      row_in_tile / rows_per_sub_block; // Sub-block within tile
  size_t row_in_sub_block =
      row_in_tile % rows_per_sub_block; // Row in sub-block

  // Decompose column position
  size_t col_tile = col / cols_per_sub_block; // Which column tile
  size_t col_in_sub_block = col % cols_per_sub_block; // Column within sub-block

  // Compute tile index and offset within tile
  size_t num_col_tiles = cuda::ceil_div(num_scale_cols, cols_per_sub_block);
  size_t tile_idx = tile_row * num_col_tiles + col_tile;

  size_t offset_in_tile =
      (row_in_sub_block * sub_blocks_per_tile * cols_per_sub_block) +
      (sub_block_row * cols_per_sub_block) + col_in_sub_block;

  constexpr size_t tile_size = rows_per_tile * cols_per_sub_block;
  return tile_idx * tile_size + offset_in_tile;
}

namespace cu {

__global__ void repack_scales(
    const uint8_t* scales_linear,
    uint8_t* scales_tiled,
    size_t input_rows,
    size_t input_cols,
    size_t output_rows,
    size_t output_cols) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x =
      cg::this_grid().dim_blocks().x * cg::this_grid().block_index().x;

  size_t output_index = tidx + grid_dim_x * size_t(tidy);
  size_t output_size = output_rows * output_cols;

  if (output_index >= output_size) {
    return;
  }

  size_t tiled_offset =
      scale_tiled_offset(output_index, output_rows, output_cols);

  size_t row = output_index / output_cols;
  size_t col = output_index % output_cols;

  // Probably this can be done better with 2 separated paths for valid and
  // padding
  if (row < input_rows && col < input_cols) {
    size_t input_index = row * input_cols + col;
    scales_tiled[tiled_offset] = scales_linear[input_index];
  } else {
    // Zero-fill padding region
    scales_tiled[tiled_offset] = 0;
  }
}

} // namespace cu

void repack_scales(
    const array& scales,
    array& scales_tiled,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(scales);
  enc.set_output_array(scales_tiled);

  // Note: scales_tiled is padded to full tiles so if num_rows or num_cols
  // are not multiples of tile sizes, the extra space is filled with zeros

  size_t input_rows = scales.shape(-2);
  size_t input_cols = scales.shape(-1);

  size_t output_rows = scales_tiled.shape(-2);
  size_t output_cols = scales_tiled.shape(-1);
  size_t output_size = output_rows * output_cols;

  bool large = output_size > UINT_MAX;
  auto [num_blocks, block_dims] = get_launch_args(
      output_size, scales_tiled.shape(), scales_tiled.strides(), large);

  enc.add_kernel_node(
      cu::repack_scales,
      num_blocks,
      block_dims,
      0,
      gpu_ptr<uint8_t>(scales),
      gpu_ptr<uint8_t>(scales_tiled),
      input_rows,
      input_cols,
      output_rows,
      output_cols);
}

} // namespace mlx::core
