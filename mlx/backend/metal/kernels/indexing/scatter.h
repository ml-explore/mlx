// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/indexing/indexing.h"

template <
    typename T,
    typename IdxT,
    typename Op,
    int NIDX,
    bool UPD_ROW_CONTIG,
    int NWORK,
    typename LocT>
METAL_FUNC void scatter_impl(
    const device T* updates,
    device mlx_atomic<T>* out,
    const constant int* upd_shape,
    const constant int64_t* upd_strides,
    const constant size_t& upd_ndim,
    const constant size_t& upd_size,
    const constant int* out_shape,
    const constant int64_t* out_strides,
    const constant size_t& out_ndim,
    const constant int* axes,
    const constant size_t& idx_size,
    const thread Indices<IdxT, NIDX>& indices,
    uint2 gid [[thread_position_in_grid]]) {
  Op op;

  auto ind_idx = gid.y * NWORK;
  LocT out_offset = 0;
  if (upd_size > 1) {
    out_offset = elem_to_loc<LocT>(
        gid.x, upd_shape + indices.ndim, out_strides, out_ndim);
  }

  for (int j = 0; j < NWORK && ind_idx < idx_size; ++j, ind_idx++) {
    LocT out_idx = out_offset;
    for (int i = 0; i < NIDX; ++i) {
      auto idx_loc = indices.row_contiguous[i]
          ? ind_idx
          : elem_to_loc<LocT>(
                ind_idx,
                &indices.shapes[indices.ndim * i],
                &indices.strides[indices.ndim * i],
                indices.ndim);
      auto ax = axes[i];
      auto idx_val = offset_neg_idx(indices.buffers[i][idx_loc], out_shape[ax]);
      out_idx +=
          static_cast<LocT>(idx_val) * static_cast<LocT>(out_strides[ax]);
    }
    auto upd_idx = ind_idx * static_cast<LocT>(upd_size) + gid.x;
    if constexpr (!UPD_ROW_CONTIG) {
      upd_idx = elem_to_loc<LocT>(upd_idx, upd_shape, upd_strides, upd_ndim);
    }
    op.atomic_update(out, updates[upd_idx], out_idx);
  }
}

template <
    typename T,
    typename IdxT,
    typename Op,
    bool OUT_ROW_CONTIG,
    bool UPD_ROW_CONTIG,
    bool UPD_SCALAR,
    int NWORK>
[[kernel]] void slice_update_op_impl(
    const device T* updates [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant int* update_shape [[buffer(2)]],
    const constant int64_t* update_strides [[buffer(3)]],
    const constant int& update_ndim [[buffer(4)]],
    const constant int64_t& update_size [[buffer(5)]],
    const constant int64_t* output_strides [[buffer(6)]],
    const constant int64_t& output_offset [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 gsize [[threads_per_grid]]) {
  Op op;

  IdxT idx = (IdxT(gid.y) * gsize.x + gid.x) * NWORK;
  IdxT out_idx;
  IdxT update_idx;

  if constexpr (OUT_ROW_CONTIG) {
    out_idx = idx;
  } else {
    out_idx = elem_to_loc<IdxT>(idx, update_shape, output_strides, update_ndim);
  }

  if constexpr (!UPD_SCALAR) {
    if constexpr (UPD_ROW_CONTIG) {
      update_idx = idx;
    } else {
      update_idx =
          elem_to_loc<IdxT>(idx, update_shape, update_strides, update_ndim);
    }
  } else {
    update_idx = 0;
  }

  out += output_offset;

  for (int j = 0; j < NWORK && idx < update_size; j++) {
    out[out_idx] = op(out[out_idx], updates[update_idx]);
    idx++;

    if constexpr (OUT_ROW_CONTIG) {
      out_idx = idx;
    } else {
      out_idx += output_strides[update_ndim - 1];
    }

    if constexpr (UPD_ROW_CONTIG) {
      update_idx = idx;
    } else {
      update_idx += update_strides[update_ndim - 1];
    }
  }
}
