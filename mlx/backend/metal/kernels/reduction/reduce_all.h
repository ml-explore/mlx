// Copyright © 2024 Apple Inc.

#pragma once

#include "mlx/backend/metal/kernels/reduction/utils.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// All reduce
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U per_thread_all_reduce(
    const device T* in,
    const device size_t& in_size,
    uint gid,
    uint grid_size) {
  Op op;
  U total_val = Op::init;

  if (gid * N_READS < in_size) {
    in += gid * N_READS;

    int r = 0;
    for (; r < (int)ceildiv(in_size, grid_size * N_READS) - 1; r++) {
      U vals[N_READS] = {op.init};

      for (int i = 0; i < N_READS; i++) {
        vals[i] = static_cast<U>(in[i]);
      }
      for (int i = 0; i < N_READS; i++) {
        total_val = op(vals[i], total_val);
      }

      in += grid_size * N_READS;
    }

    // Separate case for the last set as we close the reduction size
    size_t curr_idx = (gid + r * (size_t)grid_size) * N_READS;
    if (curr_idx < in_size) {
      int max_reads = in_size - curr_idx;
      T vals[N_READS];

      for (int i = 0, idx = 0; i < N_READS; i++, idx++) {
        idx = idx < max_reads ? idx : max_reads - 1;
        vals[i] = in[idx];
      }
      for (int i = 0; i < N_READS; i++) {
        U val = i < max_reads ? vals[i] : Op::init;
        total_val = op(static_cast<U>(val), total_val);
      }
    }
  }

  return total_val;
}