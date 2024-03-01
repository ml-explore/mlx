// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/reduction/utils.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Column reduce
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
inline U _contiguous_strided_reduce(
    const device T* in,
    threadgroup U* local_data,
    uint in_idx,
    uint reduction_size,
    uint reduction_stride,
    uint2 tid,
    uint2 lid,
    uint2 lsize) {
  Op op;
  U total_val = Op::init;

  uint base_offset = (tid.y * lsize.y + lid.y) * N_READS;
  for (uint r = 0; r < N_READS && (base_offset + r) < reduction_size; r++) {
    uint offset = base_offset + r;
    total_val =
        op(static_cast<U>(total_val), in[in_idx + offset * reduction_stride]);
  }
  local_data[lsize.y * lid.x + lid.y] = total_val;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  U val = Op::init;
  if (lid.y == 0) {
    // Perform reduction across columns in thread group
    for (uint i = 0; i < lsize.y; i++) {
      val = op(val, local_data[lsize.y * lid.x + i]);
    }
  }

  return val;
}