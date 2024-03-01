// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/reduction/utils.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Row atomics
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
inline U per_thread_row_reduce(
    const device T* in,
    const constant size_t& reduction_size,
    const constant size_t& out_size,
    const constant int* shape,
    const constant size_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x,
    uint2 tid) {
  Op op;

  // Each threadgroup handles 1 reduction
  // TODO: Specializing elem_to_loc would be slightly faster
  int idx = tid.y * out_size + tid.x;
  int extra_offset = elem_to_loc(idx, shape, strides, ndim);
  in += extra_offset + lid_x * N_READS;

  // The reduction is accumulated here
  U total_val = Op::init;

  // Loop over the reduction size within thread group
  int r = 0;
  for (; r < (int)ceildiv(reduction_size, N_READS * lsize_x) - 1; r++) {
    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      vals[i] = in[i];
    }
    for (int i = 0; i < N_READS; i++) {
      total_val = op(static_cast<U>(vals[i]), total_val);
    }

    in += lsize_x * N_READS;
  }

  // Separate case for the last set as we close the reduction size
  size_t reduction_index = (lid_x + (size_t)lsize_x * r) * N_READS;
  if (reduction_index < reduction_size) {
    int max_reads = reduction_size - reduction_index;

    T vals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      int idx = min(i, max_reads - 1);
      vals[i] = static_cast<U>(in[idx]);
    }
    for (int i = 0; i < N_READS; i++) {
      T val = i < max_reads ? vals[i] : Op::init;
      total_val = op(static_cast<U>(val), total_val);
    }
  }

  return total_val;
}