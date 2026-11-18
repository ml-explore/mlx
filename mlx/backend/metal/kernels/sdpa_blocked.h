// Copyright © 2026 Apple Inc.

#pragma once

#include <metal_stdlib>

#include "mlx/backend/metal/kernels/defines.h"

using namespace metal;

template <typename T>
[[kernel]] void sdpa_blocked_scale_copy(
    const device T* q [[buffer(0)]],
    device T* q_block [[buffer(1)]],
    constant float& scale [[buffer(2)]],
    constant int64_t* q_strides [[buffer(3)]],
    constant int& heads [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    constant int& head_dim [[buffer(6)]],
    constant int& q_offset [[buffer(7)]],
    uint3 elem [[thread_position_in_grid]]) {
  int d = elem.x;
  int row = elem.y;
  int head = elem.z % heads;
  int batch = elem.z / heads;
  int64_t q_index = int64_t(batch) * q_strides[0] +
      int64_t(head) * q_strides[1] + int64_t(q_offset + row) * q_strides[2] + d;
  size_t q_block_index = (size_t(elem.z) * rows + row) * head_dim + d;
  q_block[q_block_index] = q[q_index] * static_cast<T>(scale);
}

template <typename T>
[[kernel]] void sdpa_blocked_causal_mask(
    device T* scores [[buffer(0)]],
    constant int& tile_rows [[buffer(1)]],
    constant int& columns [[buffer(2)]],
    uint2 elem [[thread_position_in_grid]]) {
  int column = elem.x;
  int score_row = elem.y;
  int row = score_row % tile_rows;
  if (column > row) {
    size_t index = size_t(score_row) * columns + columns - tile_rows + column;
    scores[index] = Limits<T>::finite_min;
  }
}
