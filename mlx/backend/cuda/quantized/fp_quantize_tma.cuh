// Copyright © 2026 Apple Inc.
#pragma once

#include "mlx/backend/cuda/common.h"
#include "mlx/backend/cuda/ptx.cuh"
#include "mlx/backend/cuda/quantized/mxfp8_quantize.cuh"
#include "mlx/backend/cuda/quantized/nvfp4_quantize.cuh"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/backend/cuda/vector_types.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace mlx::core {
namespace cu {

constexpr size_t TMA_SHMEM_ALIGNMENT = 128;
constexpr size_t BUFFS_NUM = 2;

namespace cg = cooperative_groups;

// TMA-based rowwise quantization for MXFP8
// Input: [rows, cols] row-major
// Output: [rows, cols / 32 / group_size] row-major
// Scales: [rows, cols / group_size] - one scale per group
// Each thread processes one row of group_size elements per stage,
// Each block processes TILE_M x TILE_K elements per stage.
template <
    typename T,
    int group_size,
    bool USE_SR,
    int THREADS_PER_BLOCK,
    int ROWS_PER_BLOCK,
    int COLS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    fp_quantize_rowwise_tma_mxfp8(
        const __grid_constant__ CUtensorMap tensor_map_input,
        const __grid_constant__ CUtensorMap tensor_map_output,
        uint8_t* __restrict__ scales,
        const size_t rows,
        const size_t cols) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
  using Tx2 = Vector2_t<T>;
  using Tx4 = Vector4_t<T>;

  constexpr size_t TILE_M = ROWS_PER_BLOCK;
  constexpr size_t TILE_K = group_size;
  constexpr size_t STAGES = COLS_PER_BLOCK / group_size;

  const auto block_idx = cg::this_thread_block().group_index();
  const auto idx_in_block = cg::this_thread_block().thread_index();
  const int tidx = idx_in_block.x; // Thread handles row tidx
  const bool is_master = (tidx == 0);

  const size_t block_offset_row = block_idx.x * ROWS_PER_BLOCK;
  const size_t block_offset_col = block_idx.y * COLS_PER_BLOCK;

  constexpr size_t BUFF_ELEMS = TILE_M * TILE_K;
  constexpr size_t in_tile_size = BUFF_ELEMS * sizeof(T);
  constexpr size_t in_buff_size_aligned =
      ((in_tile_size * BUFFS_NUM + TMA_SHMEM_ALIGNMENT - 1) /
       TMA_SHMEM_ALIGNMENT) *
      TMA_SHMEM_ALIGNMENT;

  constexpr size_t out_tile_elems = BUFF_ELEMS;
  constexpr size_t out_buff_size_aligned =
      ((out_tile_elems * BUFFS_NUM + TMA_SHMEM_ALIGNMENT - 1) /
       TMA_SHMEM_ALIGNMENT) *
      TMA_SHMEM_ALIGNMENT;

  extern __shared__ char shared_mem[];
  uintptr_t aligned_shared =
      (reinterpret_cast<uintptr_t>(shared_mem) + TMA_SHMEM_ALIGNMENT - 1) &
      ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  T* in_sh = reinterpret_cast<T*>(aligned_shared);
  uint8_t* out_sh =
      reinterpret_cast<uint8_t*>(aligned_shared + in_buff_size_aligned);
  uint8_t* scales_sh = reinterpret_cast<uint8_t*>(
      aligned_shared + in_buff_size_aligned + out_buff_size_aligned);

  constexpr uint32_t tile_bytes = static_cast<uint32_t>(in_tile_size);

  __shared__ alignas(8) uint64_t mbar[STAGES]; // 8 bytes alignment for mbarrier

  T thread_data[TILE_K]; // register storage for one row of TILE_K elements
  uint32_t rbits = 0;

  const size_t groups_per_row = cols / TILE_K;

  if (is_master) {
#pragma unroll
    for (int iter = 0; iter < STAGES; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_BLOCK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();

  copy_2d_to_shared(
      &in_sh[0],
      &tensor_map_input,
      static_cast<uint32_t>(block_offset_col),
      static_cast<uint32_t>(block_offset_row),
      tile_bytes,
      &mbar[0],
      is_master);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_col_offset = stage * TILE_K;

    if (next_stage < STAGES) {
      ptx::cp_async_bulk_wait_group_read<1>();
      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_col_offset = block_offset_col + next_stage * TILE_K;
      const size_t next_buff_elem_offset = next_buff * BUFF_ELEMS;

      copy_2d_to_shared(
          &in_sh[next_buff_elem_offset],
          &tensor_map_input,
          static_cast<uint32_t>(next_col_offset),
          static_cast<uint32_t>(block_offset_row),
          tile_bytes,
          &mbar[next_stage],
          is_master);
    }

    ptx::fence_proxy_async_shared_cta();
    ptx::mbarrier_wait_parity(&mbar[stage], 0);

    const size_t buff_offset = buff * BUFF_ELEMS;
    const int lane = tidx % 32;
    constexpr int GROUPS = TILE_K / 4;
    // First made 2 naive mistakes were made:
    // 1. Stored without swizzling that results in 32 / 16 bank conflicts
    // (depending of the input type)
    // 2. More naive mistake was indexing registes with swizzling index that
    // is known only at compile time this results in massive slow down of
    // course
#pragma unroll
    for (int j = 0; j < GROUPS; ++j) {
      int swizzled_j = (j + lane) % GROUPS;
      *reinterpret_cast<Tx4*>(&thread_data[j * 4]) = *reinterpret_cast<Tx4*>(
          &in_sh[buff_offset + tidx * TILE_K + swizzled_j * 4]);
    }
    // Compute scale: find max absolute value in the row (order-independent)
    Tx2 amax_2x = Tx2{T(0.0f), T(0.0f)};
#pragma unroll
    for (int col = 0; col < TILE_K; col += 2) {
      auto pair = Tx2{thread_data[col], thread_data[col + 1]};
      absmax_x2<Tx2>(amax_2x, amax_2x, pair);
    }

    float scale =
        max(fabsf(static_cast<float>(amax_2x.x)),
            fabsf(static_cast<float>(amax_2x.y)));

    scale /= F8E4M3_MAX;

    using ScaleType = __nv_fp8_e8m0;
    auto s = ScaleType(scale);
    scale = float(s);
    scales_sh[buff * TILE_K + tidx] = s.__x;
    const size_t out_buff_offset = buff * out_tile_elems;
    // Quantize to registers first
    uint32_t quantized_regs[GROUPS];
#pragma unroll
    for (int j = 0; j < GROUPS; ++j) {
      Tx4 w_Tx4 = *reinterpret_cast<Tx4*>(&thread_data[j * 4]);
      quantized_regs[j] =
          cu::scale_cvt_Tx4_to_fp8x4<T, USE_SR>(w_Tx4, 1.0f / scale, rbits);
    }
    const int group = lane / 4; // 8 groups of 4 threads
#pragma unroll
    for (int j = 0; j < GROUPS; ++j) {
      int rotated_j = (j + group) % GROUPS;
      *reinterpret_cast<uint32_t*>(
          &out_sh[out_buff_offset + tidx * TILE_K + rotated_j * 4]) =
          quantized_regs[rotated_j];
    }
    __syncthreads();
    const size_t global_row = block_offset_row + tidx;
    const size_t global_group = (block_offset_col + stage_col_offset) / TILE_K;
    if (global_row < rows) {
      scales[global_row * groups_per_row + global_group] =
          scales_sh[buff * TILE_K + tidx];
    }
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    if (is_master) {
      const size_t global_col = block_offset_col + stage_col_offset;
      const uint32_t out_x = static_cast<uint32_t>(global_col);
      const uint32_t out_y = static_cast<uint32_t>(block_offset_row);

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t*>(&tensor_map_output),
          out_x,
          out_y,
          reinterpret_cast<uint64_t*>(&out_sh[out_buff_offset]));
      ptx::cp_async_bulk_commit_group();
    }
  }

  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  if (is_master) {
#pragma unroll
    for (int iter = 0; iter < STAGES; ++iter) {
      ptx::mbarrier_invalidate(&mbar[iter]);
    }
  }
#endif // __CUDA_ARCH__ >= 1000
}

template <
    typename T,
    bool USE_SR,
    int THREADS_PER_BLOCK,
    int COLS_PER_BLOCK,
    int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    fp_quantize_columnwise_tma_mxfp8(
        const __grid_constant__ CUtensorMap tensor_map_input,
        const __grid_constant__ CUtensorMap tensor_map_output,
        uint8_t* __restrict__ scales,
        const size_t rows,
        const size_t cols) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
  using Tx2 = Vector2_t<T>;
  using Tx4 = Vector4_t<T>;

  constexpr size_t TILE_M = 32;
  constexpr size_t TILE_K = COLS_PER_BLOCK;
  constexpr size_t STAGES = ROWS_PER_BLOCK / TILE_M;
  constexpr int elem_per_byte = 1;

  const auto block_idx = cg::this_thread_block().group_index();
  const auto idx_in_block = cg::this_thread_block().thread_index();
  const int tidx = idx_in_block.x; // Thread handles column tidx
  const bool is_master = (tidx == 0);

  const size_t block_offset_col = block_idx.x * COLS_PER_BLOCK;
  const size_t block_offset_row = block_idx.y * ROWS_PER_BLOCK;

  constexpr size_t BUFF_ELEMS = TILE_M * TILE_K;
  constexpr size_t in_tile_size = BUFF_ELEMS * sizeof(T);
  constexpr size_t in_buff_size_aligned =
      ((in_tile_size * BUFFS_NUM + TMA_SHMEM_ALIGNMENT - 1) /
       TMA_SHMEM_ALIGNMENT) *
      TMA_SHMEM_ALIGNMENT;

  constexpr size_t out_tile_elems = BUFF_ELEMS / elem_per_byte;
  constexpr size_t out_tile_size = out_tile_elems;
  constexpr size_t out_buff_size_aligned =
      ((out_tile_size * BUFFS_NUM + TMA_SHMEM_ALIGNMENT - 1) /
       TMA_SHMEM_ALIGNMENT) *
      TMA_SHMEM_ALIGNMENT;

  extern __shared__ char shared_mem[];
  uintptr_t aligned_shared =
      (reinterpret_cast<uintptr_t>(shared_mem) + TMA_SHMEM_ALIGNMENT - 1) &
      ~(static_cast<uintptr_t>(TMA_SHMEM_ALIGNMENT - 1));

  T* in_sh = reinterpret_cast<T*>(aligned_shared);
  uint8_t* out_sh =
      reinterpret_cast<uint8_t*>(aligned_shared + in_buff_size_aligned);
  uint8_t* scales_sh = reinterpret_cast<uint8_t*>(
      aligned_shared + in_buff_size_aligned + out_buff_size_aligned);

  constexpr uint32_t tile_bytes = static_cast<uint32_t>(in_tile_size);

  __shared__ alignas(8) uint64_t mbar[STAGES];

  T thread_data[TILE_M];
  uint32_t rbits = 0; // Reserved for stochastic rounding
  const size_t scale_stride = rows / TILE_M;

  // Master thread init memory barriers for all stages
  // fence for tma, synchronize threads so all see mbarrier
  if (is_master) {
#pragma unroll
    for (int iter = 0; iter < STAGES; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_BLOCK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  __syncthreads();
  // Launch first async copy before entering the loop
  copy_2d_to_shared(
      &in_sh[0],
      &tensor_map_input,
      static_cast<uint32_t>(block_offset_col),
      static_cast<uint32_t>(block_offset_row),
      tile_bytes,
      &mbar[0],
      is_master);

#pragma unroll
  for (size_t stage = 0; stage < STAGES; ++stage) {
    // buffer memory offset in shared memory (we use double buffering for
    // pipelining)
    const size_t buff = stage % BUFFS_NUM;
    const size_t next_stage = stage + 1;
    const size_t stage_row_offset = stage * TILE_M;

    if (next_stage < STAGES) {
      // before launching another async copy, check that there is less than 2
      // (to ensure that shared -> global synch is finished and buffer can be
      // reused)
      ptx::cp_async_bulk_wait_group_read<1>();
      const size_t next_buff = next_stage % BUFFS_NUM;
      const size_t next_row_offset = block_offset_row + next_stage * TILE_M;
      const size_t next_buff_elem_offset = next_buff * BUFF_ELEMS;

      copy_2d_to_shared(
          &in_sh[next_buff_elem_offset],
          &tensor_map_input,
          static_cast<uint32_t>(block_offset_col),
          static_cast<uint32_t>(next_row_offset),
          tile_bytes,
          &mbar[next_stage],
          is_master);
    }

    ptx::fence_proxy_async_shared_cta();
    // Wait until the data is ready, parity is always 0 because for simplicity
    // we dont reuse barriers between stages
    ptx::mbarrier_wait_parity(&mbar[stage], 0);
    const size_t buff_offset = buff * BUFF_ELEMS;
    // Read the data from shared to registers
#pragma unroll
    for (int row = 0; row < TILE_M; ++row) {
      thread_data[row] = in_sh[buff_offset + row * TILE_K + tidx];
    }
    Tx2 amax_2x = Tx2{T(0.0f), T(0.0f)};
#pragma unroll
    for (int row = 0; row < TILE_M; row += 2) {
      auto pair = Tx2{thread_data[row], thread_data[row + 1]};
      absmax_x2<Tx2>(amax_2x, amax_2x, pair);
    }

    float scale =
        max(fabsf(static_cast<float>(amax_2x.x)),
            fabsf(static_cast<float>(amax_2x.y)));

    scale /= F8E4M3_MAX;

    using ScaleType = __nv_fp8_e8m0;
    auto s = ScaleType(scale);
    scale = float(s);
    scales_sh[buff * TILE_K + tidx] = s.__x;
    const size_t out_buff_offset = buff * out_tile_elems;
    // Quantize to registers first
    constexpr int GROUPS = TILE_M / 4;
    uint32_t quantized_regs[GROUPS];
#pragma unroll
    for (int j = 0; j < GROUPS; ++j) {
      Tx4 w_Tx4 = *reinterpret_cast<Tx4*>(&thread_data[j * 4]);
      quantized_regs[j] =
          cu::scale_cvt_Tx4_to_fp8x4<T, USE_SR>(w_Tx4, 1.0f / scale, rbits);
    }
    // Write output to shared memory with swapped store order to reduce bank
    // conflicts. Without swap: stride between threads is TILE_M=32 bytes
    // = 8 banks, every 4th thread hits the same bank -> 8-way conflict.
    const int lane = tidx % 32;
    const int group = (lane / 4) % 2;
    const size_t base = out_buff_offset + tidx * TILE_M;
    switch (group) {
      case 0:
        *reinterpret_cast<uint4*>(&out_sh[base + 0]) = {
            quantized_regs[0],
            quantized_regs[1],
            quantized_regs[2],
            quantized_regs[3]};
        *reinterpret_cast<uint4*>(&out_sh[base + 16]) = {
            quantized_regs[4],
            quantized_regs[5],
            quantized_regs[6],
            quantized_regs[7]};
        break;
      case 1:
        *reinterpret_cast<uint4*>(&out_sh[base + 16]) = {
            quantized_regs[4],
            quantized_regs[5],
            quantized_regs[6],
            quantized_regs[7]};
        *reinterpret_cast<uint4*>(&out_sh[base + 0]) = {
            quantized_regs[0],
            quantized_regs[1],
            quantized_regs[2],
            quantized_regs[3]};
        break;
    }
    __syncthreads();
    // Thread tidx computes scale for input column (block_offset_col + tidx)
    // This scale goes to output row (block_offset_col + tidx), column
    // (global_row_group)
    const size_t global_row_group =
        (block_offset_row + stage_row_offset) / TILE_M;
    const size_t global_col = block_offset_col + tidx;
    // TODO: scale writing is not good
    if (global_col < cols && (block_offset_row + stage_row_offset) < rows) {
      scales[global_col * scale_stride + global_row_group] =
          scales_sh[buff * TILE_K + tidx];
    }
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    if (is_master) {
      const size_t global_row = block_offset_row + stage_row_offset;
      const uint32_t out_x = static_cast<uint32_t>(global_row);
      const uint32_t out_y = static_cast<uint32_t>(block_offset_col);

      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t*>(&tensor_map_output),
          out_x,
          out_y,
          reinterpret_cast<uint64_t*>(&out_sh[out_buff_offset]));
      ptx::cp_async_bulk_commit_group();
    }
  }
  // Wait for all TMA stores to complete
  ptx::cp_async_bulk_wait_group_read<0>();

  __syncthreads();
  if (is_master) {
#pragma unroll
    for (int iter = 0; iter < STAGES; ++iter) {
      ptx::mbarrier_invalidate(&mbar[iter]);
    }
  }
#endif // __CUDA_ARCH__ >= 1000
}

template <typename T, bool USE_SR>
__global__ void __launch_bounds__(128) fp_quantize_columnwise_tma_nvfp4(
    const __grid_constant__ CUtensorMap tensor_map_input,
    const __grid_constant__ CUtensorMap tensor_map_output,
    uint8_t* __restrict__ scales,
    const size_t rows,
    const size_t cols,
    float* global_scale) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000)
  // placeholder TODO - NVFP4 TMA kernel not yet implemented
#endif // __CUDA_ARCH__ >= 1000
}

} // namespace cu
} // namespace mlx::core
