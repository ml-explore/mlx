// Copyright © 2025 Apple Inc.
#pragma once

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

template <int bits>
struct Dequantize {
  __device__ float operator()(uint8_t x) {
    if constexpr (bits == 8) {
      return float(*(__nv_fp8_e4m3*)(&x));
    } else {
      return float(*(__nv_fp4_e2m1*)(&x));
    }
  }
};

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
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto s = ScaleType(scale_dec_b);
  float scale_enc_b = scale_enc / float(s);
  float scale_dec = float(s) * inv_scale_enc;
  AlignedVector<T, group_size> w_hat;

#pragma unroll
  for (int i = 0; i < group_size / 4; i++) {
    Tx4 w_Tx4 = *reinterpret_cast<Tx4*>(&w_tile[i * 4]);
    float4 dq;
    if constexpr (bits == 8) {
      uint32_t quantized_val =
          scale_cvt_Tx4_to_fp8x4<T, USE_SR>(w_Tx4, scale_enc_b, rbits);
      dq = dequant_fp8(quantized_val);
    } else {
      uint16_t quantized_val =
          scale_cvt_Tx4_to_fp4x4<T, USE_SR>(w_Tx4, scale_enc_b, rbits);
      dq = dequant_fp4(quantized_val);
    }
    w_hat[i * 4] = static_cast<T>(dq.x * scale_dec);
    w_hat[i * 4 + 1] = static_cast<T>(dq.y * scale_dec);
    w_hat[i * 4 + 2] = static_cast<T>(dq.z * scale_dec);
    w_hat[i * 4 + 3] = static_cast<T>(dq.w * scale_dec);
  }
  store_vector<group_size>(out, thread_idx, w_hat);
}

template <typename T, int group_size, int bits, bool use_mx_scale, bool USE_SR>
__global__ void fp_quantize_rowwise_fallback(
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
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto s = ScaleType(scale_dec_b);
  uint8_t q_scale = s.__x;
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
    using ScaleType =
        std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
    auto s = ScaleType(scale_dec_b);
    float scale_enc_b = scale_enc / float(s);
    scales_smem[tidx][tidy] = s.__x;

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
