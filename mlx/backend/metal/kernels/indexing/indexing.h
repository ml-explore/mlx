// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_stdlib>

template <typename IdxT, int NIDX>
struct Indices {
  const array<const device IdxT*, NIDX> buffers;
  const constant int* shapes;
  const constant int64_t* strides;
  const constant bool* row_contiguous;
  const int ndim;
};

template <typename IdxT>
METAL_FUNC size_t offset_neg_idx(IdxT idx, int size) {
  if (is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return (idx < 0) ? idx + size : idx;
  }
}

template <typename IdxT>
METAL_FUNC bool
check_bounds(IdxT idx, int size, device atomic<int32_t>* global_failure) {
  if (idx < 0 || idx >= size) {
    atomic_store_explicit(global_failure, BOUNDS_FAILURE, memory_order_relaxed);
    return false;
  }
  return true;
}
