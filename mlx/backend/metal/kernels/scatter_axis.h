// Copyright © 2025 Apple Inc.

#pragma once

template <
    typename T,
    typename IdxT,
    typename LocT,
    typename Op,
    bool UpdC,
    bool IdxC>
[[kernel]] void scatter_axis(
    const device T* upd [[buffer(0)]],
    const device IdxT* indices [[buffer(1)]],
    device mlx_atomic<T>* out [[buffer(2)]],
    const constant int* shape [[buffer(3)]],
    const constant int64_t* upd_strides [[buffer(4)]],
    const constant int64_t* idx_strides [[buffer(5)]],
    const constant size_t& ndim [[buffer(6)]],
    const constant int& axis [[buffer(7)]],
    const constant int& out_axis_size [[buffer(8)]],
    const constant size_t& upd_ax_stride [[buffer(9)]],
    const constant size_t& idx_ax_stride [[buffer(10)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  Op op;

  LocT elem_idx = index.z * static_cast<LocT>(grid_dim.x);

  LocT idx_loc = index.y * static_cast<LocT>(idx_ax_stride);
  if (IdxC) {
    idx_loc += elem_idx * grid_dim.y + index.x;
  } else {
    idx_loc += elem_to_loc<LocT>(elem_idx + index.x, shape, idx_strides, ndim);
  }

  auto idx_val = indices[idx_loc];
  if (is_signed_v<IdxT>) {
    idx_val = (idx_val < 0) ? idx_val + out_axis_size : idx_val;
  }

  LocT upd_idx = index.y * static_cast<LocT>(upd_ax_stride);
  if (UpdC) {
    upd_idx += elem_idx * grid_dim.y + index.x;
  } else {
    upd_idx += elem_to_loc<LocT>(elem_idx + index.x, shape, upd_strides, ndim);
  }

  LocT out_idx = elem_idx * static_cast<LocT>(out_axis_size) +
      idx_val * grid_dim.x + index.x;
  op.atomic_update(out, upd[upd_idx], out_idx);
}
