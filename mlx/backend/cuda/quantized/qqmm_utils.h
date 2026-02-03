// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/array.h"
#include "mlx/backend/cuda/device.h"

namespace mlx::core {

// Compute padded dimensions for tiled layout
// Tiles are 128 rows × 4 columns, must allocate full tiles
inline std::pair<int, int> get_padded_scale_dims(int num_rows, int num_cols) {
  constexpr int rows_per_tile = 128;
  constexpr int cols_per_tile = 4;

  int padded_rows =
      ((num_rows + rows_per_tile - 1) / rows_per_tile) * rows_per_tile;
  int padded_cols =
      ((num_cols + cols_per_tile - 1) / cols_per_tile) * cols_per_tile;

  return {padded_rows, padded_cols};
}

inline array pad_and_swizzle_scales(
    const array& scale,
    cu::CommandEncoder& encoder,
    const Stream& s) {
  // Compute padded dimensions for full tiles (128 rows × 4 cols)
  auto [pad_outer, pad_inner] =
      get_padded_scale_dims(scale.shape(-2), scale.shape(-1));
  // cuBLAS requirements for scale factor layout:
  // 1. Dimensions must be padded to full tiles (128 rows × 4 cols)
  // 2. Out-of-bounds values must be filled with zeros
  // 3. Starting addresses must be 16-byte aligned
  // https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
  // Note: cu::malloc_async already provides 256-byte alignment
  array scale_tiled(
      cu::malloc_async(pad_outer * pad_inner, encoder),
      Shape{pad_outer, pad_inner},
      scale.dtype());
  swizzle_scales(scale, scale_tiled, encoder, s);

  encoder.add_temporary(scale_tiled);
  return scale_tiled;
}

void swizzle_scales(
    const array& scales,
    array& scales_tiled,
    cu::CommandEncoder& enc,
    const Stream& s);

// Compute alpha = tensor_amax_x * tensor_amax_w / (448 * 6)^2
// Allocate beta zero on device as well

void compute_qqmm_pointers(
    array& alpha_out,
    array& beta_out,
    const array& tensor_amax_x,
    const array& tensor_amax_w,
    cu::CommandEncoder& enc);

} // namespace mlx::core
