// Copyright © 2023-2024 Apple Inc.

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  constexpr int n_reads = 4;
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  U totals[n_reads];
  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  IdxT column = IdxT(gid.x) * lsize.x * n_reads + lid.x * n_reads;
  if (column >= reduction_stride) {
    return;
  }
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(lid.y, reduce_shape, reduce_strides);
  for (IdxT r = lid.y; r < total_rows; r += lsize.y) {
    row = in + loop.location();
    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(static_cast<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? static_cast<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }
    loop.next(lsize.y, reduce_shape, reduce_strides);
  }

  if (lsize.y > 1) {
    // lsize.y should be <= 8
    threadgroup U shared_vals[32 * 8 * n_reads];
    for (int i = 0; i < n_reads; i++) {
      shared_vals[lid.y * lsize.x * n_reads + lid.x * n_reads + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = shared_vals[lid.x * n_reads + i];
      }
      for (uint j = 1; j < lsize.y; j++) {
        for (int i = 0; i < n_reads; i++) {
          totals[i] =
              op(shared_vals[j * lsize.x * n_reads + lid.x * n_reads + i],
                 totals[i]);
        }
      }
    }
  }

  if (lid.y == 0) {
    out += out_idx * IdxT(reduction_stride) + column;
    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        out[i] = totals[i];
      }
    } else {
      for (int i = 0; column + i < reduction_stride; i++) {
        out[i] = totals[i];
      }
    }
  }
}

template <typename T, typename U, typename Op, typename IdxT, int NDIMS>
[[kernel]] void col_reduce_longcolumn(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant size_t& out_size [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  Op op;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  IdxT out_idx = gid.x + gsize.x * IdxT(gid.y);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + lid.x;

  U total = Op::init;
  IdxT total_rows = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(gid.z * lsize.y + lid.y, reduce_shape, reduce_strides);
  for (IdxT r = gid.z * lsize.y + lid.y; r < total_rows;
       r += lsize.y * gsize.z) {
    row = in + loop.location();
    total = op(static_cast<U>(*row), total);
    loop.next(lsize.y * gsize.z, reduce_shape, reduce_strides);
  }

  threadgroup U shared_vals[32 * 32];
  shared_vals[lid.y * lsize.x + lid.x] = total;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid.y == 0) {
    for (uint i = 1; i < lsize.y; i++) {
      total = op(total, shared_vals[i * lsize.x + lid.x]);
    }
    out[gid.z * IdxT(out_size) + out_idx * IdxT(reduction_stride) + lid.x] =
        total;
  }
}

/**
 * Our approach is the following simple looped approach:
 *  1. Each thread keeps running totals for BN / n_simdgroups outputs.
 *  2. Load a tile BM, BN in registers and accumulate in the running totals
 *  3. Move ahead by BM steps until the column axis and the non column
 *     reductions are exhausted.
 *  6. If BM == 32 then transpose in SM and simd reduce the running totals.
 *     Otherwise write in shared memory and BN threads accumulate the running
 *     totals with a loop.
 *  7. Write them to the output
 */
template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y, reduce_shape, reduce_strides);
  for (IdxT r = offset.y; r < total; r += BM) {
    row = in + loop.location();

    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(static_cast<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? static_cast<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }

    loop.next(BM, reduce_shape, reduce_strides);
  }

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
  if (BM == 32) {
    constexpr int n_outputs = BN / n_simdgroups;
    static_assert(
        BM != 32 || n_outputs == n_reads,
        "The tile should be selected such that n_outputs == n_reads");
    for (int i = 0; i < n_reads; i++) {
      shared_vals[offset.y * BN + offset.x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
    for (int i = 0; i < n_outputs; i++) {
      totals[i] =
          op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
    }

    // Write the output.
    if (simd_lane_id == 0) {
      IdxT out_column = BN * gid.x + out_offset.x;
      out += out_idx * IdxT(reduction_stride) + out_column;
      if (out_column + n_outputs <= reduction_stride) {
        for (int i = 0; i < n_outputs; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; out_column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }

  // Each thread holds n_reads partial results. We write them all out to shared
  // memory and threads with offset.y == 0 aggregate the columns and write the
  // outputs.
  else {
    short x_block = offset.x / n_reads;
    for (int i = 0; i < n_reads; i++) {
      shared_vals[x_block * BM * n_reads + i * BM + offset.y] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (offset.y == 0) {
      for (int i = 0; i < n_reads; i++) {
        for (int j = 1; j < BM; j++) {
          totals[i] =
              op(shared_vals[x_block * BM * n_reads + i * BM + j], totals[i]);
        }
      }
    }

    // Write the output.
    if (offset.y == 0) {
      out += out_idx * IdxT(reduction_stride) + column;
      if (safe) {
        for (int i = 0; i < n_reads; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; column + i < reduction_stride; i++) {
          out[i] = totals[i];
        }
      }
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int BM,
    int BN>
[[kernel]] void col_reduce_2pass(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    const constant size_t& out_size [[buffer(11)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 8;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;
  constexpr int n_outputs = BN / n_simdgroups;
  constexpr short outer_blocks = 32;
  static_assert(BM == 32, "BM should be equal to 32");

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  IdxT column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  IdxT full_idx = gid.y + gsize.y * IdxT(gid.z);
  IdxT block_idx = full_idx / IdxT(out_size);
  IdxT out_idx = full_idx % IdxT(out_size);
  IdxT in_idx = elem_to_loc<IdxT>(out_idx, shape, strides, ndim);
  in += in_idx + column;

  IdxT total = IdxT(non_col_reductions) * IdxT(reduction_size);
  loop.next(offset.y + block_idx * BM, reduce_shape, reduce_strides);
  for (IdxT r = offset.y + block_idx * BM; r < total; r += outer_blocks * BM) {
    row = in + loop.location();

    if (safe) {
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(static_cast<U>(row[i]), totals[i]);
      }
    } else {
      U vals[n_reads];
      for (int i = 0; i < n_reads; i++) {
        vals[i] =
            (column + i < reduction_stride) ? static_cast<U>(row[i]) : op.init;
      }
      for (int i = 0; i < n_reads; i++) {
        totals[i] = op(vals[i], totals[i]);
      }
    }

    loop.next(outer_blocks * BM, reduce_shape, reduce_strides);
  }

  // We can use a simd reduction to accumulate across BM so each thread writes
  // the partial output to SM and then each simdgroup does BN / n_simdgroups
  // accumulations.
  for (int i = 0; i < n_reads; i++) {
    shared_vals[offset.y * BN + offset.x + i] = totals[i];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  short2 out_offset(simd_group_id * n_outputs, simd_lane_id);
  for (int i = 0; i < n_outputs; i++) {
    totals[i] =
        op.simd_reduce(shared_vals[out_offset.y * BN + out_offset.x + i]);
  }

  // Write the output.
  if (simd_lane_id == 0) {
    IdxT out_column = BN * gid.x + out_offset.x;
    out += full_idx * IdxT(reduction_stride) + out_column;
    if (out_column + n_outputs <= reduction_stride) {
      for (int i = 0; i < n_outputs; i++) {
        out[i] = totals[i];
      }
    } else {
      for (int i = 0; out_column + i < reduction_stride; i++) {
        out[i] = totals[i];
      }
    }
  }
}
