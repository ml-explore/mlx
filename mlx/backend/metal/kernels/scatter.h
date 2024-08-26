// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/indexing.h"

template <typename T, typename IdxT, typename Op, int NIDX>
METAL_FUNC void scatter_1d_index_impl(
    const device T* updates [[buffer(1)]],
    device mlx_atomic<T>* out [[buffer(2)]],
    const constant int* out_shape [[buffer(3)]],
    const constant size_t* out_strides [[buffer(4)]],
    const constant size_t& out_ndim [[buffer(5)]],
    const constant int* upd_shape [[buffer(6)]],
    const constant size_t& upd_ndim [[buffer(7)]],
    const constant size_t& upd_size [[buffer(8)]],
    const thread array<const device IdxT*, NIDX>& idx_buffers,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;

  size_t out_idx = 0;
  for (int i = 0; i < NIDX; i++) {
    auto idx_val = offset_neg_idx(idx_buffers[i][gid.y], out_shape[i]);
    out_idx += idx_val * out_strides[i];
  }

  if (upd_ndim > 1) {
    auto out_offset = elem_to_loc(gid.x, upd_shape + 1, out_strides, out_ndim);
    out_idx += out_offset;
  } else {
    out_idx += gid.x;
  }

  op.atomic_update(out, updates[gid.y * upd_size + gid.x], out_idx);
}

template <typename T, typename IdxT, typename Op, int NIDX>
METAL_FUNC void scatter_impl(
    const device T* updates [[buffer(1)]],
    device mlx_atomic<T>* out [[buffer(2)]],
    const constant int* upd_shape [[buffer(3)]],
    const constant size_t* upd_strides [[buffer(4)]],
    const constant size_t& upd_ndim [[buffer(5)]],
    const constant size_t& upd_size [[buffer(6)]],
    const constant int* out_shape [[buffer(7)]],
    const constant size_t* out_strides [[buffer(8)]],
    const constant size_t& out_ndim [[buffer(9)]],
    const constant int* axes [[buffer(10)]],
    const thread Indices<IdxT, NIDX>& indices,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;
  auto ind_idx = gid.y;
  auto ind_offset = gid.x;

  size_t out_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    auto idx_loc = elem_to_loc(
        ind_idx,
        &indices.shapes[indices.ndim * i],
        &indices.strides[indices.ndim * i],
        indices.ndim);
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], out_shape[ax]);
    out_idx += idx_val * out_strides[ax];
  }

  if (upd_size > 1) {
    auto out_offset = elem_to_loc(
        ind_offset, upd_shape + indices.ndim, out_strides, out_ndim);
    out_idx += out_offset;
  }

  auto upd_idx =
      elem_to_loc(gid.y * upd_size + gid.x, upd_shape, upd_strides, upd_ndim);
  op.atomic_update(out, updates[upd_idx], out_idx);
}
