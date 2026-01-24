// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/indexing/indexing.h"

template <typename T, typename IdxT, typename LocT, int N>
[[kernel]] void gather_front(
    const device T* src,
    const device IdxT* indices,
    device T* out,
    const constant int64_t& stride,
    const constant int& size,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {
  auto idx = offset_neg_idx(indices[index.y], size);
  LocT src_idx = static_cast<LocT>(stride) * idx;
  LocT out_idx = static_cast<LocT>(stride) * index.y;

  int s_idx = N * index.x;
  for (int i = 0; i < N && s_idx < stride; ++i, ++s_idx) {
    out[out_idx + s_idx] = src[src_idx + s_idx];
  }
}
