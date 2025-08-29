// Copyright © 2025 Apple Inc.

#pragma once

template <typename T, typename IdxT, typename LocT, bool SrcC, bool IdxC>
[[kernel]] void gather_axis(
    const device T* src [[buffer(0)]],
    const device IdxT* indices [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant int* shape [[buffer(3)]],
    const constant int64_t* src_strides [[buffer(4)]],
    const constant int64_t* idx_strides [[buffer(5)]],
    const constant size_t& ndim [[buffer(6)]],
    const constant int& axis [[buffer(7)]],
    const constant int& axis_size [[buffer(8)]],
    const constant size_t& src_ax_stride [[buffer(9)]],
    const constant size_t& idx_ax_stride [[buffer(10)]],
    uint3 index [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  LocT elem_idx = index.z * static_cast<LocT>(grid_dim.x);
  LocT out_idx = elem_idx * grid_dim.y + index.x;

  LocT idx_loc = index.y * static_cast<LocT>(idx_ax_stride);
  if (IdxC) {
    idx_loc += out_idx;
  } else {
    idx_loc += elem_to_loc<LocT>(elem_idx + index.x, shape, idx_strides, ndim);
  }

  auto idx_val = indices[idx_loc];
  if (is_signed_v<IdxT>) {
    idx_val = (idx_val < 0) ? idx_val + axis_size : idx_val;
  }

  LocT src_idx = idx_val * static_cast<LocT>(src_ax_stride);
  if (SrcC) {
    src_idx += elem_idx * axis_size + index.x;
  } else {
    src_idx += elem_to_loc<LocT>(elem_idx + index.x, shape, src_strides, ndim);
  }

  out_idx += index.y * static_cast<LocT>(grid_dim.x);
  out[out_idx] = src[src_idx];
}
