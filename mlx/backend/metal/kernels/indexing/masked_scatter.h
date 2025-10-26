// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/indexing/indexing.h"

template <typename T, bool mask_batched, bool src_batched>
METAL_FUNC void masked_assign_impl(
    const device bool* mask,
    const device uint* scatter_offsets,
    const device T* src,
    device T* out,
    const thread Indices<bool, 1>& idxs_mask,
    const thread Indices<T, 1>& idxs_src,
    const thread Indices<T, 1>& idxs_out,
    uint src_size,
    uint inner,
    uint src_block,
    uint src_capacity,
    uint idx) {
  const int mask_ndim = idxs_mask.ndim;
  const constant int* mask_shape = idxs_mask.shapes;
  const constant long* mask_strides = idxs_mask.strides;

  bool mask_value;
  if (mask_ndim == 0) {
    mask_value = static_cast<bool>(mask[0]);
  } else {
    const long mask_loc = elem_to_loc<long>(
        static_cast<long>(idx), mask_shape, mask_strides, mask_ndim);
    mask_value = static_cast<bool>(mask[static_cast<size_t>(mask_loc)]);
  }
  if (!mask_value) {
    return;
  }

  uint src_index = scatter_offsets[idx];
  if constexpr (mask_batched) {
    if (inner == 0) {
      return;
    }
    const uint batch_idx = idx / inner;
    const uint batch_offset = batch_idx * inner;
    src_index -= scatter_offsets[batch_offset];
    if constexpr (src_batched) {
      if (src_index >= src_block) {
        return;
      }
      src_index += batch_idx * src_block;
    } else {
      if (src_index >= src_capacity) {
        return;
      }
    }
  }

  if (src_index >= src_size) {
    // TODO: Log error
    return;
  }

  const int src_ndim = idxs_src.ndim;
  const constant int* src_shape = idxs_src.shapes;
  const constant long* src_strides = idxs_src.strides;

  size_t src_loc = 0;
  if (src_ndim != 0) {
    src_loc = static_cast<size_t>(elem_to_loc<long>(
        static_cast<long>(src_index), src_shape, src_strides, src_ndim));
  }

  const T value = src[src_loc];

  const int out_ndim = idxs_out.ndim;
  const constant int* out_shape = idxs_out.shapes;
  const constant long* out_strides = idxs_out.strides;

  if (out_ndim == 0) {
    out[0] = value;
    return;
  }

  const size_t out_loc = static_cast<size_t>(elem_to_loc<long>(
      static_cast<long>(idx), out_shape, out_strides, out_ndim));
  out[out_loc] = value;
}

template <typename T, bool mask_batched, bool src_batched>
[[kernel]] void masked_assign(
    const device bool* mask [[buffer(0)]],
    const device uint* scatter_offsets [[buffer(1)]],
    const device T* src [[buffer(2)]],
    device T* out [[buffer(3)]],
    const constant int* mask_shape [[buffer(4)]],
    const constant long* mask_strides [[buffer(5)]],
    const constant int& mask_ndim [[buffer(6)]],
    const constant int* src_shape [[buffer(7)]],
    const constant long* src_strides [[buffer(8)]],
    const constant int& src_ndim [[buffer(9)]],
    const constant int* out_shape [[buffer(10)]],
    const constant long* out_strides [[buffer(11)]],
    const constant int& out_ndim [[buffer(12)]],
    const constant uint& src_size [[buffer(13)]],
    const constant uint& total [[buffer(14)]],
    const constant uint& inner [[buffer(15)]],
    const constant uint& src_block [[buffer(16)]],
    const constant uint& src_capacity [[buffer(17)]],
    uint thread_index [[thread_position_in_grid]]) {
  if (thread_index >= total) {
    // TODO: Log errors
    return;
  }

  const Indices<bool, 1> idxs_mask{
      {{mask}}, mask_shape, mask_strides, nullptr, mask_ndim};
  const Indices<T, 1> idxs_src{
      {{src}}, src_shape, src_strides, nullptr, src_ndim};
  const Indices<T, 1> idxs_out{
      {{out}}, out_shape, out_strides, nullptr, out_ndim};

  masked_assign_impl<T, mask_batched, src_batched>(
      mask,
      scatter_offsets,
      src,
      out,
      idxs_mask,
      idxs_src,
      idxs_out,
      src_size,
      inner,
      src_block,
      src_capacity,
      thread_index);
}
