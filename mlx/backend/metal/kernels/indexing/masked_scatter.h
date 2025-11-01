// Copyright © 2024 Apple Inc.

#pragma once

template <typename T>
[[kernel]] void masked_assign_impl(
    const device bool* mask [[buffer(0)]],
    const device uint* scatter_offsets [[buffer(1)]],
    const device T* src [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int* mask_shapes [[buffer(4)]],
    const constant long* mask_strides [[buffer(5)]],
    const constant int& mask_ndim [[buffer(6)]],
    const constant int* src_shapes [[buffer(7)]],
    const constant long* src_strides [[buffer(8)]],
    const constant int& src_ndim [[buffer(9)]],
    const constant int* out_shapes [[buffer(10)]],
    const constant long* out_strides [[buffer(11)]],
    const constant int& out_ndim [[buffer(12)]],
    const constant int* offsets_shapes [[buffer(13)]],
    const constant long* offsets_strides [[buffer(14)]],
    const constant int& offsets_ndim [[buffer(15)]],
    const constant long& src_batch_size [[buffer(16)]],
    uint idx [[thread_position_in_grid]]) {
  const bool mask_value =
      mask[elem_to_loc<uint>(idx, mask_shapes, mask_strides, mask_ndim)];

  if (!mask_value) {
    return;
  }

  const uint src_index = scatter_offsets[elem_to_loc<uint>(
      idx, offsets_shapes, offsets_strides, offsets_ndim)];

  if (src_index >= src_batch_size) {
    // TODO: Log error
    return;
  }

  const uint batch_idx = idx / mask_shapes[1];
  out[elem_to_loc<uint>(idx, out_shapes, out_strides, out_ndim)] =
      src[elem_to_loc<uint>(
          batch_idx * src_batch_size + src_index,
          src_shapes,
          src_strides,
          src_ndim)];
}
