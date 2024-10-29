// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/indexing.h"

template <typename T, typename IdxT, typename Op, int NIDX, int NWORK>
METAL_FUNC void scatter_1d_index_impl(
    const device T* updates,
    device mlx_atomic<T>* out,
    const constant int* out_shape,
    const constant size_t* out_strides,
    const constant size_t& out_ndim,
    const constant int* upd_shape,
    const constant size_t& upd_ndim,
    const constant size_t& upd_size,
    const constant size_t& idx_size,
    const thread array<const device IdxT*, NIDX>& idx_buffers,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;

  size_t out_offset;
  if (upd_ndim > 1) {
    out_offset = elem_to_loc(gid.x, upd_shape + 1, out_strides, out_ndim);
  } else {
    out_offset = gid.x;
  }

  auto ind_idx = gid.y * NWORK;
  for (int j = 0; j < NWORK && ind_idx < idx_size; ++j, ind_idx++) {
    size_t out_idx = out_offset;
    for (int i = 0; i < NIDX; i++) {
      auto idx_val = offset_neg_idx(idx_buffers[i][ind_idx], out_shape[i]);
      out_idx += idx_val * out_strides[i];
    }
    op.atomic_update(out, updates[ind_idx * upd_size + gid.x], out_idx);
  }
}

template <
    typename T,
    typename IdxT,
    typename Op,
    int NIDX,
    bool UPD_ROW_CONTIG,
    int NWORK>
METAL_FUNC void scatter_impl(
    const device T* updates,
    device mlx_atomic<T>* out,
    const constant int* upd_shape,
    const constant size_t* upd_strides,
    const constant size_t& upd_ndim,
    const constant size_t& upd_size,
    const constant int* out_shape,
    const constant size_t* out_strides,
    const constant size_t& out_ndim,
    const constant int* axes,
    const constant size_t& idx_size,
    const thread Indices<IdxT, NIDX>& indices,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;

  auto ind_idx = gid.y * NWORK;
  size_t out_offset = 0;
  if (upd_size > 1) {
    out_offset =
        elem_to_loc(gid.x, upd_shape + indices.ndim, out_strides, out_ndim);
  }

  for (int j = 0; j < NWORK && ind_idx < idx_size; ++j, ind_idx++) {
    size_t out_idx = out_offset;
    for (int i = 0; i < NIDX; ++i) {
      auto idx_loc = indices.row_contiguous[i]
          ? ind_idx
          : elem_to_loc(
                ind_idx,
                &indices.shapes[indices.ndim * i],
                &indices.strides[indices.ndim * i],
                indices.ndim);
      auto ax = axes[i];
      auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], out_shape[ax]);
      out_idx += idx_val * out_strides[ax];
    }
    auto upd_idx = ind_idx * upd_size + gid.x;
    if constexpr (!UPD_ROW_CONTIG) {
      upd_idx = elem_to_loc(upd_idx, upd_shape, upd_strides, upd_ndim);
    }
    op.atomic_update(out, updates[upd_idx], out_idx);
  }
}
