// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/indexing.h"

template <typename T, typename IdxT, int NIDX, int IDX_NDIM>
METAL_FUNC void gather_impl(
    const device T* src [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant int* src_shape [[buffer(2)]],
    const constant size_t* src_strides [[buffer(3)]],
    const constant size_t& src_ndim [[buffer(4)]],
    const constant int* slice_sizes [[buffer(5)]],
    const constant int* axes [[buffer(6)]],
    const thread Indices<IdxT, NIDX>& indices,
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  size_t src_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    size_t idx_loc;
    if (IDX_NDIM == 0) {
      idx_loc = 0;
    } else if (IDX_NDIM == 1) {
      idx_loc = index.x * indices.strides[indices.ndim * i];
    } else {
      idx_loc = index.x * indices.strides[indices.ndim * i];
      idx_loc += elem_to_loc(
          index.y,
          &indices.shapes[indices.ndim * i + 1],
          &indices.strides[indices.ndim * i + 1],
          indices.ndim - 1);
    }
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], src_shape[ax]);
    src_idx += idx_val * src_strides[ax];
  }

  auto src_offset = elem_to_loc(index.z, slice_sizes, src_strides, src_ndim);

  size_t out_idx = index.z;
  if (IDX_NDIM == 1) {
    out_idx += static_cast<size_t>(grid_dim.z) * index.x;
  } else if (IDX_NDIM >= 2) {
    out_idx +=
        grid_dim.z * (index.x * static_cast<size_t>(grid_dim.y) + index.y);
  }
  out[out_idx] = src[src_offset + src_idx];
}
