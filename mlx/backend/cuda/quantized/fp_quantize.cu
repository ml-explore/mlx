// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/fp_quantize.cuh"
#include "mlx/backend/cuda/quantized/fp_quantize_tma.cuh"
#include "mlx/backend/cuda/quantized/quantized.h"
#include "mlx/backend/cuda/quantized/quantized_utils.cuh"
#include "mlx/backend/cuda/vector_types.cuh"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <iostream>

namespace mlx::core {
namespace cu {

template <
    int TILE_M,
    int TILE_K,
    int THREADS_PER_BLOCK,
    int STAGES,
    int SCALES_PER_STAGE>
inline std::tuple<dim3, dim3, size_t> get_tma_launch_args(
    size_t grid_dim_x_size, // rows
    size_t grid_dim_y_size, // cols
    size_t block_size_x, // ROWS_PER_BLOCK
    size_t block_size_y, // COL_PER_BLOCK
    int in_size_bytes, // itemsize
    int bits) {
  dim3 grid;
  grid.x = (grid_dim_x_size + block_size_x - 1) / block_size_x;
  grid.y = (grid_dim_y_size + block_size_y - 1) / block_size_y;
  grid.z = 1;

  dim3 block(THREADS_PER_BLOCK, 1, 1);

  const int elem_per_byte = bits == 8 ? 1 : 2;
  constexpr size_t BUFF_ELEMS = TILE_M * TILE_K;
  const size_t in_tile_size = BUFF_ELEMS * in_size_bytes;
  const size_t in_buff_size_aligned =
      ((in_tile_size * BUFFS_NUM + TMA_SHMEM_ALIGNMENT - 1) /
       TMA_SHMEM_ALIGNMENT) *
      TMA_SHMEM_ALIGNMENT;

  const size_t out_tile_elems = BUFF_ELEMS / elem_per_byte;
  const size_t out_buff_size_aligned =
      ((out_tile_elems * BUFFS_NUM + TMA_SHMEM_ALIGNMENT - 1) /
       TMA_SHMEM_ALIGNMENT) *
      TMA_SHMEM_ALIGNMENT;
  const size_t smem_size =
      in_buff_size_aligned + out_buff_size_aligned + TMA_SHMEM_ALIGNMENT;
  return std::make_tuple(grid, block, smem_size);
}

inline CUtensorMapDataType get_tma_dtype(Dtype dtype) {
  switch (dtype) {
    case float16:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case bfloat16:
      return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case float32:
      return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    default:
      throw std::runtime_error(
          "[fp_quantize_columnwise_tma] Unsupported dtype for TMA");
  }
}

inline std::tuple<dim3, dim3>
get_columnwise_launch_args(size_t size, int group_size, int M, int K) {
  constexpr int BLOCK_X = 16;
  constexpr int BLOCK_Y = 32;
  int rows_per_block = BLOCK_X;
  int cols_per_block = BLOCK_Y * group_size;

  dim3 grid;
  grid.x =
      cuda::ceil_div(K, cols_per_block) * cuda::ceil_div(M, rows_per_block);
  grid.y = 1;
  grid.z = 1;

  dim3 block(BLOCK_X, BLOCK_Y);

  return std::make_tuple(grid, block);
}

} // namespace cu

void fp_quantize_dequantize(
    const array& w,
    array& what,
    int group_size,
    int bits,
    const std::optional<array>& global_scale /* = std::nullopt */,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(w);
  if (global_scale.has_value()) {
    enc.set_input_array(global_scale.value());
  }
  enc.set_output_array(what);
  dispatch_float_types(w.dtype(), "fp_quantize_dequantize", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_quantize_dequantize<T, 32, 4, true, false>;
      if (bits == 8) {
        kernel = cu::fp_quantize_dequantize<T, 32, 8, true, false>;
      } else if (group_size == 16) {
        kernel = cu::fp_quantize_dequantize<T, 16, 4, false, false>;
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
          gpu_ptr<T>(what),
          w.size(),
          global_scale.has_value() ? gpu_ptr<float>(global_scale.value())
                                   : nullptr);
    }
  });
}

void fp_quantize_rowwise(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    const std::optional<array>& global_scale /* = std::nullopt */,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(w);
  if (global_scale.has_value()) {
    enc.set_input_array(global_scale.value());
  }
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  dispatch_float_types(w.dtype(), "fp_quantize_rowwise", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto kernel = cu::fp_quantize_rowwise<T, 32, 4, true, false>;
      if (bits == 8) {
        kernel = cu::fp_quantize_rowwise<T, 32, 8, true, false>;
      } else if (group_size == 16) {
        kernel = cu::fp_quantize_rowwise<T, 16, 4, false, false>;
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
          w.size(),
          global_scale.has_value() ? gpu_ptr<float>(global_scale.value())
                                   : nullptr);
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not quantize input with type float64.");
    }
  });
}

void fp_quantize_columnwise(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    const std::optional<array>& global_scale /* = std::nullopt */,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(w);
  if (global_scale.has_value()) {
    enc.set_input_array(global_scale.value());
  }
  enc.set_output_array(wq);
  enc.set_output_array(scales);
  dispatch_float_types(w.dtype(), "fp_quantize_columnwise", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      auto M = w.shape(-2);
      auto K = w.shape(-1);
      auto kernel = cu::fp_quantize_columnwise<T, 32, 4, true, false>;
      if (bits == 8) {
        kernel = cu::fp_quantize_columnwise<T, 32, 8, true, false>;
      } else if (group_size == 16) {
        kernel = cu::fp_quantize_columnwise<T, 16, 4, false, false>;
      }
      auto [num_blocks, block_dims] =
          cu::get_columnwise_launch_args(w.size(), group_size, M, K);
      enc.add_kernel_node(
          kernel,
          num_blocks,
          block_dims,
          0,
          gpu_ptr<T>(w),
          gpu_ptr<uint8_t>(wq),
          gpu_ptr<uint8_t>(scales),
          w.size(),
          M,
          K,
          global_scale.has_value() ? gpu_ptr<float>(global_scale.value())
                                   : nullptr);
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not quantize input with type float64.");
    }
  });
}

void fp_quantize_columnwise_tma(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    const std::optional<array>& global_scale /* = std::nullopt */,
    cu::CommandEncoder& enc,
    const Stream& s) {
  enc.set_input_array(w);
  enc.set_output_array(wq);
  enc.set_output_array(scales);

  size_t rows = w.shape(-1);
  size_t cols = w.size() / rows;
  size_t stride_bytes = w.strides(-1) * w.itemsize();

  // 16 bytes is minimum for TMA tile size along contigious dimension
  // For nvfp4 (group_size = 16), we can't have a tile = group_size because we
  // write transposed output so we will do 2 passes on the input tile
  if (bits == 8 && group_size == 32) {
    dispatch_float_types(
        w.dtype(), "fp_quantize_columnwise_mxfp8", [&](auto type_tag) {
          using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
          if constexpr (!std::is_same_v<T, double>) {
            constexpr int THREADS_PER_BLOCK = 64;
            constexpr int ROWS_PER_BLOCK = 64;
            constexpr int COLS_PER_BLOCK = 64;
            constexpr size_t TILE_M = 32;
            constexpr size_t TILE_K = COLS_PER_BLOCK;
            constexpr size_t STAGES = ROWS_PER_BLOCK / TILE_M;

            // For columnwise: grid.x = cols, grid.y = rows
            // scales_per_stage = TILE_K (one scale per column per stage)
            auto [grid, block, smem_size] = cu::get_tma_launch_args<
                TILE_M,
                TILE_K,
                THREADS_PER_BLOCK,
                STAGES,
                TILE_K>(
                cols, rows, COLS_PER_BLOCK, ROWS_PER_BLOCK, w.itemsize(), bits);

            CUtensorMap tensor_map_input;
            CUtensorMap tensor_map_output;

            create_2D_tensor_map(
                &tensor_map_input,
                const_cast<void*>(static_cast<const void*>(gpu_ptr<T>(w))),
                cu::get_tma_dtype(w.dtype()),
                rows,
                cols,
                TILE_M,
                TILE_K,
                stride_bytes);

            create_2D_tensor_map(
                &tensor_map_output,
                gpu_ptr<uint8_t>(wq),
                CU_TENSOR_MAP_DATA_TYPE_UINT8,
                cols,
                rows,
                TILE_K,
                TILE_M,
                rows);

            auto kernel = cu::fp_quantize_columnwise_tma_mxfp8<
                T,
                false,
                THREADS_PER_BLOCK,
                COLS_PER_BLOCK,
                ROWS_PER_BLOCK>;
            enc.add_kernel_node(
                kernel,
                grid,
                block,
                static_cast<uint32_t>(smem_size),
                tensor_map_input,
                tensor_map_output,
                gpu_ptr<uint8_t>(scales),
                rows,
                cols);
          } else {
            throw std::runtime_error(
                "[fp_quantize_columnwise_tma] Cannot quantize input with type float64.");
          }
        });
  } else {
    // NVFP4 or other configurations - TMA kernel not implemented yet
    throw std::runtime_error(
        "[fp_quantize_columnwise_tma] TMA quantization only implemented for bits=8 and group_size=32.");
  }
}

void fp_quantize_columnwise(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    const std::optional<array>& global_scale /* = std::nullopt */,
    cu::CommandEncoder& enc,
    const Stream& s) {
  // Use TMA version for SM100+ with MXFP8 (bits=8, group_size=32)
  // NVFP4 todo
  const size_t rows = w.shape(-1);
  const size_t cols = w.size() / rows;
  const bool has_full_tma_tiles = ((rows % 128) == 0) && ((cols % 128) == 0);
  bool use_tma =
      (enc.device().compute_capability_major() >= 10 && bits == 8 &&
       group_size == 32 && has_full_tma_tiles);
  if (use_tma) {
    fp_quantize_columnwise_tma(
        w, wq, scales, group_size, bits, global_scale, enc, s);
  } else {
    fp_quantize_columnwise(
        w, wq, scales, group_size, bits, global_scale, enc, s);
  }
}

void fp_quantize(
    const array& w,
    array& wq,
    array& scales,
    int group_size,
    int bits,
    const std::optional<array>& global_scale /* = std::nullopt */,
    cu::CommandEncoder& enc,
    const Stream& s) {
  if (w.strides(-1) == 1) {
    fp_quantize_rowwise(w, wq, scales, group_size, bits, global_scale, enc, s);
  } else {
    fp_quantize_columnwise(
        w, wq, scales, group_size, bits, global_scale, enc, s);
  }
}

void fp_dequantize(
    const array& wq,
    const array& scales,
    array& w,
    int group_size,
    int bits,
    const std::optional<array>& global_scale /* = std::nullopt */,
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
  if (global_scale.has_value()) {
    enc.set_input_array(global_scale.value());
  }
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
          w.size(),
          global_scale.has_value() ? gpu_ptr<float>(global_scale.value())
                                   : nullptr);
    } else {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Can not dequantize to output with type float64.");
    }
  });
}

} // namespace mlx::core
