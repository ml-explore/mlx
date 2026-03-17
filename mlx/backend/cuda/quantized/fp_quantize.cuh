// Copyright © 2025 Apple Inc.
#pragma once

#include "mlx/backend/cuda/ptx.cuh"
#include "mlx/backend/cuda/quantized/mxfp8_quantize.cuh"
#include "mlx/backend/cuda/quantized/nvfp4_quantize.cuh"
#include "mlx/backend/cuda/vector_types.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>

constexpr float F8E4M3_MAX = 448.0f;
constexpr float F4E2M1_MAX = 6.0f;
constexpr size_t TMA_SHMEM_ALIGNMENT = 128;
constexpr size_t BUFFS_NUM = 2;

namespace mlx::core {
namespace cu {

template <int bits>
struct Dequantize {
  __device__ float operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return float(*(cutlass::float_e4m3_t*)(&x));
    } else {
      return float(*(cutlass::float_e2m1_t*)(&x));
    }
  }
};

template <typename T>
__device__ __forceinline__ void absmax_x2(T& out, const T& x1, const T& x2) {
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

__device__ __forceinline__ void copy_2d_to_shared(
    void* dst,
    const CUtensorMap* tensor_map,
    uint32_t tile_x,
    uint32_t tile_y,
    uint32_t num_bytes,
    uint64_t* barrier,
    const bool is_master_thread) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  if (is_master_thread) {
    // Arrive and tell how many bytes are expected
    ptx::mbarrier_arrive_expect_tx(barrier, num_bytes);
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        dst, tensor_map, tile_x, tile_y, barrier);
  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(barrier);
  }
#endif // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

namespace cg = cooperative_groups;

template <typename T, int group_size, int bits, bool use_mx_scale, bool USE_SR>
__global__ void fp_quantize_dequantize(
    T* w,
    T* out,
    size_t size,
    float* global_scale = nullptr) {
  const bool use_global_scale = global_scale != nullptr;
  const float scale_enc =
      use_global_scale ? (F8E4M3_MAX * F4E2M1_MAX) / *global_scale : 1.0f;
  const float inv_scale_enc = use_global_scale ? 1.0f / scale_enc : 1.0f;

  using Tx2 = Vector2_t<T>;
  uint32_t rbits = 0; // reserved bits for future use
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();
  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;
  auto grid_dim_x = cg::this_grid().dim_blocks().x * block_size.x;

  size_t thread_idx = tidx + grid_dim_x * size_t(tidy);
  size_t base_idx = thread_idx * group_size;

  if (base_idx >= size) {
    return;
  }

  auto w_tile = load_vector<group_size, T>(w, thread_idx);
  float scale_dec_b = 0.0f;

  Tx2 amax_2x = Tx2{0.0f, 0.0f};

#pragma unroll
  for (int i = 0; i < group_size; i += 2) {
    auto pair = Tx2{w_tile[i], w_tile[i + 1]};
    absmax_x2<Tx2>(amax_2x, amax_2x, pair);
  }

  scale_dec_b = static_cast<float>(
      max(fabsf(static_cast<float>(amax_2x.x)),
          fabsf(static_cast<float>(amax_2x.y))));

  scale_dec_b /= bits == 4 ? F4E2M1_MAX : F8E4M3_MAX;
  scale_dec_b *= scale_enc;
  // Convert to mx scale or nv scale
  using ScaleType = std::conditional_t<
      use_mx_scale,
      cutlass::float_ue8m0_t,
      cutlass::float_e4m3_t>;
  auto s = ScaleType(scale_dec_b);
  float scale_enc_b = scale_enc / float(s);
  float scale_dec = float(s) * inv_scale_enc;
  AlignedVector<T, group_size> w_hat;

#pragma unroll
  for (int i = 0; i < group_size / 8; i++) {
    auto& w = *reinterpret_cast<cutlass::Array<T, 8>*>(&w_tile[i * 8]);
    cutlass::NumericArrayConverter<float, T, 8> fp32_t;
    auto scaled = fp32_t(w) * scale_enc_b;
    cutlass::Array<float, 8> dq;
    if constexpr (bits == 8) {
      cutlass::NumericArrayConverter<cutlass::float_e4m3_t, float, 8> fp8_fp32;
      auto quant = fp8_fp32(scaled);
      cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, 8> fp32_fp8;
      dq = fp32_fp8(quant);
    } else {
      cutlass::NumericArrayConverter<cutlass::float_e2m1_t, float, 8> fp4_fp32;
      auto quant = fp4_fp32(scaled);
      cutlass::NumericArrayConverter<float, cutlass::float_e2m1_t, 8> fp32_fp4;
      dq = fp32_fp4(quant);
    }
    cutlass::NumericArrayConverter<T, float, 8> t_fp32;
    *reinterpret_cast<cutlass::Array<T, 8>*>(&w_hat[i * 8]) =
        t_fp32(dq * scale_dec);
  }
  store_vector<group_size>(out, thread_idx, w_hat);
}

template <typename T, int group_size, int bits, bool use_mx_scale, bool USE_SR>
__global__ void fp_quantize_rowwise(
    T* w,
    uint8_t* out,
    uint8_t* scales,
    size_t size,
    float* global_scale = nullptr) {
  // NVFP4 conversion:
  // Global encode scale: (448 × 6) / *global_scale
  // Per-block decode scale: S_dec_b = (block_amax / 6) × S_enc → stored as FP8
  // E4M3 Per-block encode scale: S_enc_b = S_enc / S_dec_b
  const bool use_global_scale = global_scale != nullptr;
  const float scale_enc =
      use_global_scale ? (F8E4M3_MAX * F4E2M1_MAX) / *global_scale : 1.0f;

  using Tx2 = Vector2_t<T>;
  using Tx4 = Vector4_t<T>;
  uint32_t rbits = 0; // reserved bits for future use
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();
  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;
  auto grid_dim_x = cg::this_grid().dim_blocks().x * block_size.x;

  size_t thread_idx = tidx + grid_dim_x * size_t(tidy);
  size_t base_idx = thread_idx * group_size;

  if (base_idx >= size) {
    return;
  }

  auto w_tile = load_vector<group_size, T>(w, thread_idx);
  float scale_dec_b = 0.0f;

  Tx2 amax_2x = Tx2{0.0f, 0.0f};

#pragma unroll
  for (int i = 0; i < group_size; i += 2) {
    auto pair = Tx2{w_tile[i], w_tile[i + 1]};
    absmax_x2<Tx2>(amax_2x, amax_2x, pair);
  }

  scale_dec_b = static_cast<float>(
      max(fabsf(static_cast<float>(amax_2x.x)),
          fabsf(static_cast<float>(amax_2x.y))));

  scale_dec_b /= bits == 4 ? F4E2M1_MAX : F8E4M3_MAX;
  scale_dec_b *= scale_enc;
  // Convert to mx scale or nv scale
  using ScaleType = std::conditional_t<
      use_mx_scale,
      cutlass::float_ue8m0_t,
      cutlass::float_e4m3_t>;
  auto s = ScaleType(scale_dec_b);
  uint8_t q_scale = s.storage;
  float scale_enc_b = scale_enc / float(s);

  scales[thread_idx] = q_scale;
  constexpr int elem_per_byte = bits == 8 ? 1 : 2;
  AlignedVector<uint8_t, group_size / elem_per_byte> quantized;

#pragma unroll
  for (int i = 0; i < group_size / 4; i++) {
    Tx4 w_Tx4 = *reinterpret_cast<Tx4*>(&w_tile[i * 4]);
    if constexpr (bits == 8) {
      uint32_t quantized_val =
          scale_cvt_Tx4_to_fp8x4<T, USE_SR>(w_Tx4, scale_enc_b, rbits);
      *reinterpret_cast<uint32_t*>(&quantized[i * 4]) = quantized_val;
    } else {
      uint16_t quantized_val =
          scale_cvt_Tx4_to_fp4x4<T, USE_SR>(w_Tx4, scale_enc_b, rbits);
      *reinterpret_cast<uint16_t*>(&quantized[i * 2]) = quantized_val;
    }
  }
  store_vector<group_size / elem_per_byte>(out, thread_idx, quantized);
}

template <
    typename T,
    bool USE_SR,
    int THREADS_PER_BLOCK,
    int COLS_PER_BLOCK,
    int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    fp_quantize_columnwise_mxfp8(
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
  constexpr size_t STEPS = ROWS_PER_BLOCK / TILE_M;
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

  constexpr uint32_t tile_bytes = static_cast<uint32_t>(in_tile_size);

  __shared__ alignas(8) uint64_t mbar[STEPS];

  T thread_data[TILE_M];
  uint32_t rbits = 0; // Reserved for stochastic rounding
  const size_t scale_stride = rows / TILE_M;

  // Master thread init memory barriers for all steps
  // fence for tma, synchronize threads so all see mbarrier
  if (is_master) {
#pragma unroll
    for (int iter = 0; iter < STEPS; ++iter) {
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
  for (size_t step = 0; step < STEPS; ++step) {
    // buffer memory offset in shared memory (we use double buffering for
    // pipelining)
    const size_t buff = step % BUFFS_NUM;
    const size_t next_step = step + 1;
    const size_t step_row_offset = step * TILE_M;

    if (next_step < STEPS) {
      // before launching another async copy, check that there is less than 2
      // (to ensure that shared -> global synch is finished and buffer can be
      // reused)
      ptx::cp_async_bulk_wait_group_read<1>();
      const size_t next_buff = next_step % BUFFS_NUM;
      const size_t next_row_offset = block_offset_row + next_step * TILE_M;
      const size_t next_buff_elem_offset = next_buff * BUFF_ELEMS;

      copy_2d_to_shared(
          &in_sh[next_buff_elem_offset],
          &tensor_map_input,
          static_cast<uint32_t>(block_offset_col),
          static_cast<uint32_t>(next_row_offset),
          tile_bytes,
          &mbar[next_step],
          is_master);
    }

    ptx::fence_proxy_async_shared_cta();
    // Wait until the data is ready, parity is always 0 because for simplicity
    // we dont reuse barriers between steps
    ptx::mbarrier_wait_parity(&mbar[step], 0);
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
    // Write scale directly to global memory
    const size_t global_col = block_offset_col + tidx;
    const size_t global_row_group =
        (block_offset_row + step_row_offset) / TILE_M;
    if (global_col < cols && (block_offset_row + step_row_offset) < rows) {
      scales[global_col * scale_stride + global_row_group] = s.__x;
    }
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
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();

    if (is_master) {
      const size_t global_row = block_offset_row + step_row_offset;
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
    for (int iter = 0; iter < STEPS; ++iter) {
      ptx::mbarrier_invalidate(&mbar[iter]);
    }
  }
#endif // __CUDA_ARCH__ >= 1000
}

// TODO: add kernel with tma instructions
template <typename T, int group_size, int bits, bool use_mx_scale, bool USE_SR>
__global__ void fp_quantize_columnwise_fallback(
    T* w,
    uint8_t* out,
    uint8_t* scales,
    size_t size,
    int M,
    int K,
    float* global_scale = nullptr) {
  // Input: [M, K] with strides [1, M] (M-major)
  // Quantized output: [M, K/elem_per_byte] row-major (K-major)
  // Scales: [M, K/group_size] row-major (K-major)
  // Quantize along K (last dimension, groups of group_size elements)
  const bool use_global_scale = global_scale != nullptr;
  const float scale_enc =
      use_global_scale ? (F8E4M3_MAX * F4E2M1_MAX) / *global_scale : 1.0f;

  using Tx2 = Vector2_t<T>;
  using Tx4 = Vector4_t<T>;
  uint32_t rbits = 0;

  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  constexpr int BLOCK_X = 16;
  constexpr int BLOCK_Y = 32;
  constexpr int elem_per_byte = (bits == 8) ? 1 : 2;
  constexpr int bytes_per_group = group_size / elem_per_byte;

  constexpr int rows_per_block = BLOCK_X;
  constexpr int cols_per_block = BLOCK_Y * group_size;
  constexpr int local_cols = cols_per_block / elem_per_byte;
  constexpr int bytes_per_block = rows_per_block * local_cols;

  constexpr int SMEM_PAD = 4;
  constexpr int padded_local_cols = local_cols + SMEM_PAD;

  auto tidx = idx_in_block.x;
  auto tidy = idx_in_block.y;

  int num_col_blocks = (K + cols_per_block - 1) / cols_per_block;
  auto bidx = block_idx.x % num_col_blocks;
  auto bidy = block_idx.x / num_col_blocks;

  T thread_data[group_size];

  __shared__ uint8_t quantized_smem[rows_per_block * padded_local_cols];
  __shared__ uint8_t scales_smem[BLOCK_X][BLOCK_Y + SMEM_PAD];

  int row_base = bidy * rows_per_block + tidx;
  int col_base = bidx * cols_per_block + tidy * group_size;

  bool valid = (row_base < M) && (col_base + group_size <= K);
  if (valid) {
#pragma unroll
    for (int i = 0; i < group_size; i++) {
      auto index = row_base + (col_base + i) * M;
      thread_data[i] = w[index];
    }

    // Compute scale
    Tx2 amax_2x = Tx2{0.0f, 0.0f};
#pragma unroll
    for (int r = 0; r < group_size; r += 2) {
      auto pair = Tx2{thread_data[r], thread_data[r + 1]};
      absmax_x2<Tx2>(amax_2x, amax_2x, pair);
    }
    float scale_dec_b =
        max(fabsf(static_cast<float>(amax_2x.x)),
            fabsf(static_cast<float>(amax_2x.y)));
    scale_dec_b /= bits == 4 ? F4E2M1_MAX : F8E4M3_MAX;
    scale_dec_b *= scale_enc;
    // Convert to mx scale or nv scale
    using ScaleType = std::conditional_t<
        use_mx_scale,
        cutlass::float_ue8m0_t,
        cutlass::float_e4m3_t>;
    auto s = ScaleType(scale_dec_b);
    float scale_enc_b = scale_enc / float(s);
    scales_smem[tidx][tidy] = s.storage;

    int shared_idx = tidx * padded_local_cols + tidy * bytes_per_group;

#pragma unroll
    for (int j = 0; j < group_size / 4; j++) {
      Tx4 w_Tx4 = *reinterpret_cast<Tx4*>(&thread_data[j * 4]);
      if constexpr (bits == 8) {
        uint32_t quantized_val =
            scale_cvt_Tx4_to_fp8x4<T, USE_SR>(w_Tx4, scale_enc_b, rbits);
        *reinterpret_cast<uint32_t*>(&quantized_smem[shared_idx + j * 4]) =
            quantized_val;
      } else {
        uint16_t quantized_val =
            scale_cvt_Tx4_to_fp4x4<T, USE_SR>(w_Tx4, scale_enc_b, rbits);
        *reinterpret_cast<uint16_t*>(&quantized_smem[shared_idx + j * 2]) =
            quantized_val;
      }
    }
  }
  __syncthreads();

  int output_cols = K / elem_per_byte;
  int num_groups_per_row = K / group_size;
  int linear_tid = tidx + tidy * BLOCK_X;
  // Write back quantized values
#pragma unroll
  for (int i = linear_tid; i < bytes_per_block; i += BLOCK_X * BLOCK_Y) {
    int local_row = i / local_cols;
    int local_col = i % local_cols;

    int global_row = bidy * rows_per_block + local_row;
    int global_col = bidx * local_cols + local_col;

    if (global_row < M && global_col < output_cols) {
      int physical_idx = local_row * padded_local_cols + local_col;
      out[global_row * output_cols + global_col] = quantized_smem[physical_idx];
    }
  }
  // Write back scales
  constexpr int num_scales = BLOCK_X * BLOCK_Y;
#pragma unroll
  for (int i = linear_tid; i < num_scales; i += BLOCK_X * BLOCK_Y) {
    int local_row = i / BLOCK_Y;
    int local_col = i % BLOCK_Y;

    int global_row = bidy * BLOCK_X + local_row;
    int global_col = bidx * BLOCK_Y + local_col;

    if (global_row < M && global_col < num_groups_per_row) {
      scales[global_row * num_groups_per_row + global_col] =
          scales_smem[local_row][local_col];
    }
  }
}

template <typename T, int group_size, int bits, bool use_mx_scale>
__global__ void fp_dequantize(
    const uint8_t* w,
    const uint8_t* scales,
    T* out,
    size_t size,
    float* global_scale = nullptr) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x = cg::this_grid().dim_blocks().x * block_size.x;

  constexpr int pack_factor = bits == 8 ? 1 : 2;
  const bool use_global_scale = global_scale != nullptr;
  const float inv_scale_enc = use_mx_scale
      ? 1.0f
      : (use_global_scale ? (*global_scale) / (F8E4M3_MAX * F4E2M1_MAX) : 1.0f);
  size_t offset = tidx + grid_dim_x * size_t(tidy);
  size_t oindex = offset * pack_factor;

  if (oindex >= size) {
    return;
  }

  size_t gindex = oindex / group_size;
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto scale = float(((ScaleType*)(scales))[gindex]) * inv_scale_enc;

  out += oindex;

  uint32_t val = w[offset];
#pragma clang loop unroll(full)
  for (int i = 0; i < pack_factor; i++) {
    uint8_t d;
    if (bits == 4) {
      d = (val >> (bits * i)) & 0x0f;
    } else if (bits == 8) {
      d = val;
    }
    out[i] = static_cast<T>(scale * Dequantize<bits>{}(d));
  }
}

} // namespace cu
} // namespace mlx::core
