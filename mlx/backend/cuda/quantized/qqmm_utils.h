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

void repack_scales(
    const array& scales,
    array& scales_tiled,
    cu::CommandEncoder& enc,
    const Stream& s);

} // namespace mlx::core
