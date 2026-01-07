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

inline std::tuple<dim3, dim3> get_swizzle_launch_args(
    size_t M_swizzled,
    size_t K_swizzled,
    int tile_rows = 128,
    int tile_cols = 4,
    int tiles_per_lane = 1) {
  constexpr int lanes_per_block = 32 // 32 threads per warp
      int lanes_per_block =
          tiles_per_lane* lanes_per_block // tiles_per_lane = 1 if load int, 4
                                          // if int4
      const int tiles_per_block = lanes_per_block * tiles_per_lane;
  const int warps_per_block = tile_rows / 4; // 128 / 4 = 32

  const int num_tiles_k = K_swizzled / tile_cols;
  const int num_tiles_m = M_swizzled / tile_rows;

  dim3 grid;
  grid.x = cuda::ceil_div(num_tiles_k, tiles_per_block);
  grid.y = num_tiles_m;
  grid.z = 1;

  // Block is always (32, 32) = 1024 threads
  dim3 block(lanes_per_block, warps_per_block, 1);

  return std::make_tuple(grid, block);
}

namespace cu {

__global__ void swizzle_scaling_factors_impl(
    const uint8_t* scales_linear,
    uint8_t* scales_swizzled,
    const size_t M,
    const size_t K,
    const size_t M_swizzled,
    const size_t K_swizzled) {
  // M_swizzled and K_swizzled are dimensions of scales_tiled array
  // (padded to full tiles 128x4 if M or K are not multiples of tile sizes)

  // Tile dimensions for scale factors
  constexpr int tile_dim_row = 128;
  constexpr int tile_dim_col = 4;
  constexpr int tile_size = tile_dim_row * tile_dim_col; // 512 bytes
  constexpr int num_tile_rows_per_thread = 4; // always 4
  constexpr int num_tiles_per_thread = 1;
  constexpr int int4_per_tile =
      tile_dim_row* tile_dim_col / 16 constexpr int lanes_per_block = 32;
  constexpr int num_tiles_per_block = lanes_per_block * num_tiles_per_thread;
  // Each thread loads 4 rows of 4 bytes x 1 column of scales (16 bytes -- 16
  // scales) thread (0, 0) loads scales at rows 0,32,64,96 of tile 0 thread (1,
  // 0) loads rows 0,32,64,96 of of tile 1 therefore a warp loads: consecutive 4
  // bytes x 32 = 128 bytes 4 times with a stride 32 this can be improved
  // further with processing 2/4 tiles within a warp: load 8/16 bytes but it
  // will be a bit more complicated, because then we would need to shuffle
  // registers so for now each thread process exactly 1 tile

  auto block_size = cg::this_thread_block().dim_threads(); // (32, 32, 1)
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = idx_in_block.x; // lane within warp (0, 31)
  auto tidy = idx_in_block.y; // warp index (0, 31)
  auto linear_tid = tidy * block_size.x + tidx;

  const int bid_x = block_idx.x;
  const int bid_y = block_idx.y;

  // [M tile * 128 * bytes] + [K tile * 32]
  const int K_int = K_swizzled / 4;
  // incase of overflow cast to size_t
  const size_t input_block_offset =
      static_cast<size_t>(bid_y) * tile_dim_row * K_int +
      static_cast<size_t>(bid_x) * num_tiles_per_block;
  const int* input_block =
      reinterpret_cast<const int*>(scales_linear) + input_block_offset;

  const size_t output_offset =
      static_cast<size_t>(bid_y) * tile_dim_row * K_int +
      static_cast<size_t>(bid_x) * num_tiles_per_block * tile_size / 4;

  int* output_block = reinterpret_cast<int*>(scales_swizzled) + output_offset;

  const int num_tiles_k = K_swizzled / tile_dim_col;
  const int grid_dim_x = cg::this_grid().dim_blocks().x;
  const int grid_dim_y = cg::this_grid().dim_blocks().y;

  // each thread store 16 consecutive scales, but this store is strided within a
  // warp so we will write first to the shared memory (each thread 4 int32 -- 4
  // rows -- 16 scales) then it will be coaleased store shared -> global within
  // a thread block
  extern __shared__ int4 strided_scales_thread[];
  // load
  int thread_tile_rows[num_tile_rows_per_thread];
#pragma unroll
  for (int i = 0; i < num_tile_rows_per_thread; i++) {
    const int thread_offset =
        (i * block_size.x + tidy) * K_int + tidx * num_tiles_per_thread;
    thread_tile_rows[i] = __ldg(input_block + thread_offset);
  }
  // todo add padding  + boundaries
  //  write 4 ints to the shared memory
  strided_scales_thread[tidx * tile_size / 16 + tidy] =
      *reinterpret_cast<int4*>(thread_tile_rows);
  __syncthreads();

  // shared -> global
  __align__(16) int4* output_block_int4 = reinterpret_cast<int4*>(output_block);
  const int total_int4s = tiles_in_block * tile_size / 16;
#pragma unroll
  for (int i = linear_tid; i < total_int4s; i += block_size.x * block_size.y) {
    output_block_int4[i] = strided_scales_thread[i];
  }
}
} // namespace cu

void swizzle_scaling_factors(
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

  uint32_t smem_bytes = 128 * 4 * 32;
  auto [num_blocks, block_dims] =
      get_swizzle_launch_args(output_rows, output_cols);

  enc.add_kernel_node(
      cu::swizzle_scaling_factors_impl,
      num_blocks,
      block_dims,
      smem_bytes,
      gpu_ptr<uint8_t>(scales),
      gpu_ptr<uint8_t>(scales_tiled),
      input_rows,
      input_cols,
      output_rows,
      output_cols);
}

} // namespace mlx::core
