// Copyright © 2025 Apple Inc.

#pragma once

template <typename T, bool src_contiguous>
[[kernel]] void masked_assign_impl(
    const device bool* mask [[buffer(0)]],
    const device uint* scatter_offsets [[buffer(1)]],
    const device T* src [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int* src_shapes [[buffer(4)]],
    const constant int64_t* src_strides [[buffer(5)]],
    const constant int& src_ndim [[buffer(6)]],
    const constant int64_t& src_batch_size [[buffer(7)]],
    const constant int64_t& mask_batch_size [[buffer(8)]],
    uint idx [[thread_position_in_grid]]) {
  const bool mask_value = mask[idx];
  if (!mask_value) {
    return;
  }

  const uint src_index = scatter_offsets[idx];
  if (src_index >= src_batch_size) {
    return;
  }

  const uint batch_idx = idx / mask_batch_size;

  if (src_contiguous) {
    out[idx] = src[batch_idx * src_batch_size + src_index];
  } else {
    out[idx] = src[elem_to_loc<uint>(
        batch_idx * src_batch_size + src_index,
        src_shapes,
        src_strides,
        src_ndim)];
  }
}
