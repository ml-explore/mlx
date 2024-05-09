// Copyright © 2023-2024 Apple Inc.

#pragma once

#include <metal_stdlib>

template <typename IdxT, int NIDX>
struct Indices {
  const array<const device IdxT*, NIDX> buffers;
  const constant int* shapes;
  const constant size_t* strides;
  const int ndim;
};

template <typename IdxT>
METAL_FUNC size_t offset_neg_idx(IdxT idx, size_t size) {
  if (is_unsigned_v<IdxT>) {
    return idx;
  } else {
    return (idx < 0) ? idx + size : idx;
  }
}
