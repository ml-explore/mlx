// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qqmm_utils.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cg = cooperative_groups;

constexpr int TILE_ROWS = 128;
constexpr int TILE_COLS = 4;
constexpr int TILES_PER_LANE = 1;
constexpr int LANES_PER_BLOCK = 32;

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

inline std::tuple<dim3, dim3> get_swizzle_launch_args(
    size_t M_swizzled,
    size_t K_swizzled) {
  constexpr int tiles_per_block = LANES_PER_BLOCK * TILES_PER_LANE;
  constexpr int warps_per_block = TILE_ROWS / 4; // 128 / 4 = 32

  const int num_tiles_k = K_swizzled / TILE_COLS;
  const int num_tiles_m = M_swizzled / TILE_ROWS;

  dim3 grid;
  grid.x = cuda::ceil_div(num_tiles_k, tiles_per_block);
  grid.y = num_tiles_m;
  grid.z = 1;
  // Block is always (32, 32) = 1024 threads
  dim3 block(LANES_PER_BLOCK, warps_per_block, 1);

  return std::make_tuple(grid, block);
}

namespace cu {

__global__ void swizzle_scales(
    const uint8_t* scales_linear,
    uint8_t* scales_swizzled,
    const size_t M,
    const size_t K,
    const size_t M_swizzled,
    const size_t K_swizzled) {
  constexpr int tile_size = TILE_ROWS * TILE_COLS;
  constexpr int num_tile_rows_per_thread = 4;
  constexpr int max_tiles_per_block = LANES_PER_BLOCK * TILES_PER_LANE;

  constexpr int tile_stride = tile_size / 16; // 32 int4s per tile

  // Each thread loads 16 scales from 4 rows (stride 32) and packs them into
  // int4. For example: thread (0, 0) loads scales at rows 0,32,64,96 of tile 0,
  // thread (1, 0) loads rows 0,32,64,96 of of tile 1, etc.
  // The store is strided within a warp (stride 32 int4s), so we first
  // write to shared memory, then do a coalesced store from shared to global
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = idx_in_block.x;
  auto tidy = idx_in_block.y;
  auto linear_tid = tidy * block_size.x + tidx;

  const int bid_x = block_idx.x;
  const int bid_y = block_idx.y;

  const int K_int = K_swizzled / 4;

  const size_t output_offset = static_cast<size_t>(bid_y) * TILE_ROWS * K_int +
      static_cast<size_t>(bid_x) * max_tiles_per_block * tile_size / 4;
  int* output_block = reinterpret_cast<int*>(scales_swizzled) + output_offset;

  const int grid_dim_x = cg::this_grid().dim_blocks().x;
  const int grid_dim_y = cg::this_grid().dim_blocks().y;

  int remaining = K_int - bid_x * max_tiles_per_block;
  int tiles_in_block = min(remaining, max_tiles_per_block);
  bool valid_tile = tidx * TILES_PER_LANE < tiles_in_block;

  __shared__ int4 strided_scales_thread[max_tiles_per_block * tile_stride];

  // Initialize to zero for padding
  int thread_tile_rows[num_tile_rows_per_thread] = {0};

  if (valid_tile) {
    const size_t col_base =
        static_cast<size_t>(bid_x) * max_tiles_per_block * TILE_COLS +
        tidx * TILE_COLS;

    const bool aligned_k = (K % 4 == 0);

    if (aligned_k) {
      // fast path: K is aligned, use vectorized loads with stride K/4
      const int K_stride = K / 4;
      const size_t block_offset =
          static_cast<size_t>(bid_y) * TILE_ROWS * K_stride +
          static_cast<size_t>(bid_x) * max_tiles_per_block;
      const int* input_block =
          reinterpret_cast<const int*>(scales_linear) + block_offset;
// load
#pragma unroll
      for (int i = 0; i < num_tile_rows_per_thread; i++) {
        const size_t row =
            static_cast<size_t>(bid_y) * TILE_ROWS + i * block_size.x + tidy;
        const int thread_offset =
            (i * block_size.x + tidy) * K_stride + tidx * TILES_PER_LANE;
        if (row < M && col_base + TILE_COLS <= K) {
          thread_tile_rows[i] = __ldg(input_block + thread_offset);
        } else if (row < M) {
// partial tile at K boundary: load byte-by-byte
#pragma unroll
          for (int c = 0; c < TILE_COLS; c++) {
            if (col_base + c < K) {
              reinterpret_cast<uint8_t*>(&thread_tile_rows[i])[c] =
                  scales_linear[row * K + col_base + c];
            }
          }
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < num_tile_rows_per_thread; i++) {
        const size_t row =
            static_cast<size_t>(bid_y) * TILE_ROWS + i * block_size.x + tidy;
        if (row < M) {
          const size_t row_start = row * K;
#pragma unroll
          for (int c = 0; c < TILE_COLS; c++) {
            if (col_base + c < K) {
              reinterpret_cast<uint8_t*>(&thread_tile_rows[i])[c] =
                  scales_linear[row_start + col_base + c];
            }
          }
        }
      }
    }
    // store to shared with XOR swizzle to avoid bank conflicts
    int base_idx = tidx * tile_stride + tidy;
    int xor_bits = (tidy >> 3) & 0x3;
    int swizzled_idx = base_idx ^ xor_bits;
    strided_scales_thread[swizzled_idx] =
        *reinterpret_cast<int4*>(thread_tile_rows);
  }

  cg::thread_block block = cg::this_thread_block();
  cg::sync(block);

  const int total_int4s = tiles_in_block * tile_stride;
#pragma unroll
  for (int i = linear_tid; i < total_int4s; i += block_size.x * block_size.y) {
    int tile_idx = i / tile_stride;
    int row_idx = i % tile_stride;
    int base_idx = tile_idx * tile_stride + row_idx;
    int xor_bits = (row_idx >> 3) & 0x3;
    int swizzled_idx = base_idx ^ xor_bits;
    reinterpret_cast<int4*>(output_block)[i] =
        strided_scales_thread[swizzled_idx];
  }
}
} // namespace cu

void swizzle_scales(
    const array& scales,
    array& scales_tiled,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(scales);
  enc.set_output_array(scales_tiled);
  // Note: scales_tiled is padded to full tiles so if num_rows or num_cols
  // are not multiples of tile sizes
  size_t input_rows = scales.shape(-2);
  size_t input_cols = scales.shape(-1);

  size_t output_rows = scales_tiled.shape(-2);
  size_t output_cols = scales_tiled.shape(-1);

  auto [num_blocks, block_dims] =
      get_swizzle_launch_args(output_rows, output_cols);
  enc.add_kernel_node(
      cu::swizzle_scales,
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
