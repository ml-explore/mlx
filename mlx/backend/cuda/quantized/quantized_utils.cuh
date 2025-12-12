// Copyright © 2025 Apple Inc.

namespace mlx::core {

namespace cu {

template <int bits, int wsize = 8>
inline constexpr __device__ short get_pack_factor() {
  return (bits == 3 || bits == 5) ? 8 : (bits == 6 ? 4 : wsize / bits);
}

template <int bits, int wsize = 8>
inline constexpr __device__ short get_bytes_per_pack() {
  constexpr int power_of_2_bits = (bits & (bits - 1)) == 0;
  return power_of_2_bits ? (wsize / 8) : (bits == 5 ? 5 : 3);
}

template <typename T>
__device__ __forceinline__ void abs_max_x2(T& out, const T& x1, const T& x2) {
  if constexpr (
      (std::is_same<T, __nv_bfloat162>::value) ||
      (std::is_same<T, __half2>::value)) {
    T a = x1;
    T b = x2;
    out = __hmax2(__habs2(a), __habs2(b));
  } else if constexpr (std::is_same<T, float2>::value) {
    float2 a = x1;
    float2 b = x2;
    out.x = fmaxf(fabsf(a.x), fabsf(b.x));
    out.y = fmaxf(fabsf(a.y), fabsf(b.y));
  }
}

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

__device__ size_t scale_inverse_tiled_offset(
    size_t tiled_offset,
    size_t num_rows,
    size_t num_scale_cols) {
  // Compute the inverse of the tiled layout offset for scale factors
  // This function maps from the tiled layout offset back to the linear scale
  // index
  //
  // Input: tiled layout offset (as produced by scale_tiled_offset)
  // Output: original linear scale index (0 to num_rows * num_scale_cols - 1)
  //
  // This reverses the tiling transformation done for tensor cores.

  constexpr size_t rows_per_tile = 128;
  constexpr size_t rows_per_sub_block = 32;
  constexpr size_t cols_per_sub_block = 4;
  constexpr size_t sub_blocks_per_tile = 4; // Vertically stacked
  constexpr size_t tile_size = rows_per_tile * cols_per_sub_block;

  // Step 1: Extract tile index and offset within tile
  size_t tile_idx = tiled_offset / tile_size;
  size_t offset_in_tile = tiled_offset % tile_size;

  // Step 2: Reverse the within-tile offset calculation
  // Original: offset_in_tile = (row_in_sub_block * 16) + (sub_block_row * 4) +
  // col_in_sub_block
  size_t row_in_sub_block =
      offset_in_tile / (sub_blocks_per_tile * cols_per_sub_block);
  size_t remainder =
      offset_in_tile % (sub_blocks_per_tile * cols_per_sub_block);

  size_t sub_block_row = remainder / cols_per_sub_block;
  size_t col_in_sub_block = remainder % cols_per_sub_block;

  // Step 3: Reconstruct row_in_tile from sub-block components
  size_t row_in_tile = (sub_block_row * rows_per_sub_block) + row_in_sub_block;

  // Step 4: Decompose tile_idx back to tile_row and col_tile
  size_t num_col_tiles = cuda::ceil_div(num_scale_cols, cols_per_sub_block);
  size_t tile_row = tile_idx / num_col_tiles;
  size_t col_tile = tile_idx % num_col_tiles;

  // Step 5: Reconstruct global row and column coordinates
  size_t row = (tile_row * rows_per_tile) + row_in_tile;
  size_t col = (col_tile * cols_per_sub_block) + col_in_sub_block;

  // Step 6: Convert back to linear index
  return row * num_scale_cols + col;
}

} // namespace cu

template <typename F>
void dispatch_groups(int group_size, F&& f) {
  switch (group_size) {
    case 32:
      f(std::integral_constant<int, 32>{});
      break;
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    case 128:
      f(std::integral_constant<int, 128>{});
      break;
  }
}

template <typename F>
void dispatch_bits(int bits, F&& f) {
  switch (bits) {
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 3:
      f(std::integral_constant<int, 3>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
    case 5:
      f(std::integral_constant<int, 5>{});
      break;
    case 6:
      f(std::integral_constant<int, 6>{});
      break;
    case 8:
      f(std::integral_constant<int, 8>{});
      break;
  }
}

} // namespace mlx::core
