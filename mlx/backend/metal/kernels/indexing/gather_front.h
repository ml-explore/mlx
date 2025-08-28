// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/indexing/indexing.h"

template <typename T, typename IdxT, typename LocT>
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

  out[out_idx + index.x] = src[src_idx + index.x];
}
