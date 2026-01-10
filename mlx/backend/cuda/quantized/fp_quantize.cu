// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/mxfp8_quantize.cuh"
#include "mlx/backend/cuda/quantized/nvfp4_quantize.cuh"
#include "mlx/backend/cuda/quantized/quantized.h"
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
__global__ void fp_quantize(T* w, uint8_t* out, uint8_t* scales, size_t size) {
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
  float scale = 0.0f;

  Tx2 amax_2x = Tx2{0.0f, 0.0f};

#pragma unroll
  for (int i = 0; i < group_size; i += 2) {
    auto pair = Tx2{w_tile[i], w_tile[i + 1]};
    abs_max_x2<Tx2>(amax_2x, amax_2x, pair);
  }

  scale = static_cast<float>(
      max(fabsf(static_cast<float>(amax_2x.x)),
          fabsf(static_cast<float>(amax_2x.y))));

  scale /= bits == 4 ? 6.0f : 448.0f;
  // Convert to mx scale or nv scale
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto s = ScaleType(scale);
  uint8_t q_scale = s.__x;
  scale = float(s);

  scales[thread_idx] = q_scale;
  constexpr int elem_per_byte = bits == 8 ? 1 : 2;
  AlignedVector<uint8_t, group_size / elem_per_byte> quantized;

#pragma unroll
  for (int i = 0; i < group_size / 4; i++) {
    Tx4 w_Tx4 = *reinterpret_cast<Tx4*>(&w_tile[i * 4]);
    if constexpr (bits == 8) {
      uint32_t quantized_val =
          scale_cvt_Tx4_to_fp8x4<T, USE_SR>(w_Tx4, 1.0f / scale, rbits);
      *reinterpret_cast<uint32_t*>(&quantized[i * 4]) = quantized_val;
    } else {
      uint16_t quantized_val =
          scale_cvt_Tx4_to_fp4x4<T, USE_SR>(w_Tx4, 1.0f / scale, rbits);
      *reinterpret_cast<uint16_t*>(&quantized[i * 2]) = quantized_val;
    }
  }
  store_vector<group_size / elem_per_byte>(out, thread_idx, quantized);
}

template <typename T, int group_size, int bits, bool use_mx_scale, bool USE_SR>
__global__ void fp_quantize_columnwise_transpose(
    T* w,
    uint8_t* out,
    uint8_t* scales,
    size_t size,
    int M,
    int K) {
  using Tx2 = Vector2_t<T>;
  using Tx4 = Vector4_t<T>;
  uint32_t rbits = 0; // reserved bits for future use
  // w is row major, we want to quantize columwise
  // we want to have coalesed load. For this we need to orginise threads
  // in a warp to load 128 consecutive bytes in one memory transaction
  // so each thread will load 4 bytes with stride num_rows * sizeof(T) bytes
  // group_size times to registers
  // then we will compute amax, quantize values and store scales and values in
  // shared memory finally we will write shared memory to global memory in
  // coalesced way to transpose data

  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();
  // since we need to know the size for shared memory at compile time,
  // I hardcoded the block_size.
  // It is very much unikely that we will aucnh less than 1024 threads
  constexpr int BLOCK_X = 32;
  constexpr int BLOCK_Y = 32;
  constexpr int cols_per_thread = 1; // each thread process 1 column
  constexpr int rows_per_thread = group_size;
  constexpr int cols_per_block = BLOCK_X * cols_per_thread;
  constexpr int rows_per_block = BLOCK_Y * rows_per_thread;
  constexpr int K_LT = K / cols_per_thread;

  auto tidx = idx_in_block.x; // index within block in x
  auto tidy = idx_in_block.y; // index within block in y
  auto bidx = block_idx.x % (K / cols_per_block);
  auto bidy = block_idx.x / (K / cols_per_block);

  T thread_column[rows_per_thread];

  __shared__ uint8_t scales_block[BLOCK_Y * cols_per_block];
  __shared__ uint8_t quantized_block[rows_per_block * cols_per_block];

  auto row_base = bidy * rows_per_block + tidy * rows_per_thread;
  auto col_base = bidx * cols_per_block + tidx * cols_per_thread;

  bool valid = (row_base < M) && (col_base < K);
  if (valid) {
    // load to registers
    for (int i = 0; i < rows_per_thread; i++) {
      auto index = (row_base + i) * K + col_base;
      thread_column[i] = w[index];
    }
    // find scale and store in shared memory
    Tx2 amax_2x = Tx2{0.0f, 0.0f};
    // compute amax for each column
    for (int r = 0; r < group_size; r += 2) {
      // thread_column_T[r * cols_per_thread + c]
      auto pair = Tx2{thread_column[r], thread_column[r + 1]};
      abs_max_x2<Tx2>(amax_2x, amax_2x, pair);
    }
    float scale =
        max(fabsf(static_cast<float>(amax_2x.x)),
            fabsf(static_cast<float>(amax_2x.y)));
    scale /= (bits == 4) ? 6.0f : 448.0f;
    using ScaleType =
        std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
    auto s = ScaleType(scale);

    int num_groups_per_col = M / group_size;
    int scale_col = bidy * BLOCK_Y + tidy;
    int scale_row = bidx * cols_per_block + tidx;

    scales[scale_row * num_groups_per_col + scale_col] = s.__x;

    int shared_idx = tidx * rows_per_block + tidy * group_size;
#pragma unroll
    for (int j = 0; j < group_size / 4; j++) {
      Tx4 w_Tx4 = *reinterpret_cast<Tx4*>(&thread_column[j * 4]);
      if constexpr (bits == 8) {
        uint32_t quantized_val =
            scale_cvt_Tx4_to_fp8x4<T, USE_SR>(w_Tx4, 1.0f / scale, rbits);
        *reinterpret_cast<uint32_t*>(&quantized_block[shared_idx + j * 4]) =
            quantized_val;
      } else {
        uint16_t quantized_val =
            scale_cvt_Tx4_to_fp4x4<T, USE_SR>(w_Tx4, 1.0f / scale, rbits);
        *reinterpret_cast<uint16_t*>(&quantized_block[shared_idx + j * 2]) =
            quantized_val;
      }
    }
  }
  __syncthreads();
  // coalesed store from shared to global
  int linear_tid = tidx + tidy * BLOCK_X; // 0 to 1023
  int total_elements = rows_per_block * cols_per_block;

  for (int i = linear_tid; i < total_elements; i += BLOCK_X * BLOCK_Y) {
    // Shared memory is [cols, rows], so:
    int local_col = i / rows_per_block;
    int local_row = i % rows_per_block;

    int global_col = bidx * cols_per_block + local_col;
    int global_row = bidy * rows_per_block + local_row;

    if (global_col < K && global_row < M) {
      out[global_col * M + global_row] = quantized_block[i];
    }
  }
}

template <typename T, int group_size, int bits, bool use_mx_scale>
__global__ void
fp_dequantize(const uint8_t* w, const uint8_t* scales, T* out, size_t size) {
  auto block_size = cg::this_thread_block().dim_threads();
  auto block_idx = cg::this_thread_block().group_index();
  auto idx_in_block = cg::this_thread_block().thread_index();

  auto tidx = block_idx.x * block_size.x + idx_in_block.x;
  auto tidy = block_idx.y * block_size.y + idx_in_block.y;

  auto grid_dim_x = cg::this_grid().dim_blocks().x * block_size.x;

  constexpr int pack_factor = bits == 8 ? 1 : 2;
  size_t offset = tidx + grid_dim_x * size_t(tidy);
  size_t oindex = offset * pack_factor;

  if (oindex >= size) {
    return;
  }

  size_t gindex = oindex / group_size;
  using ScaleType =
      std::conditional_t<use_mx_scale, __nv_fp8_e8m0, __nv_fp8_e4m3>;
  auto scale = float(((ScaleType*)(scales))[gindex]);

  out += oindex;

  uint val = w[offset];
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

void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  dispatch_float_types(w.dtype(), "fp_quantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_quantize<T, 32, 4, true, false>;
      if (bits == 8) {
        kernel = cu::fp_quantize<T, 32, 8, true, false>;
      } else if (group_size == 16) {
        kernel = cu::fp_quantize<T, 16, 4, false, false>;
      }
      bool large = w.size() > UINT_MAX;
      auto [num_blocks, block_dims] =
          get_launch_args(w.size(), w.shape(), w.strides(), large, group_size);

      enc.add_kernel_node(
          kernel,
          num_blocks,
          block_dims,
          0,
          gpu_ptr<T>(w),
          gpu_ptr<uint8_t>(wq),
          gpu_ptr<uint8_t>(scales),
          w.size());
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not quantize input with type float64.");
    }
  });
}

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    cu::CommandEncoder& enc,
    const Stream& s) {
  constexpr int uint8_per_uint32 = 4;
  int packs_per_int = 8 / bits;

  size_t size = w.size() / packs_per_int;
  bool large = size > UINT_MAX;
  auto grid_shape = w.shape();
  grid_shape.back() *= uint8_per_uint32;

  enc.set_input_array(wq);
  enc.set_input_array(scales);
  enc.set_output_array(w);
  dispatch_float_types(w.dtype(), "fp_dequantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_dequantize<T, 32, 4, true>;
      if (bits == 8) {
        kernel = cu::fp_dequantize<T, 32, 8, true>;
      } else if (group_size == 16) {
        kernel = cu::fp_dequantize<T, 16, 4, false>;
      }
      auto [num_blocks, block_dims] =
          get_launch_args(size, grid_shape, w.strides(), large);
      enc.add_kernel_node(
          kernel,
          num_blocks,
          block_dims,
          0,
          gpu_ptr<uint8_t>(wq),
          gpu_ptr<uint8_t>(scales),
          gpu_ptr<T>(w),
          w.size());
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not dequantize to output with type float64.");
    }
  });
}

} // namespace mlx::core
