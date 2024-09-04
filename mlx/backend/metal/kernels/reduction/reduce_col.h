// Copyright © 2023-2024 Apple Inc.

template <
    typename T,
    typename U,
    typename Op,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;
  looped_elem_to_loc<NDIMS> loop;
  const device T* row;

  // Case 1: Small row small column
  if (reduction_size * non_col_reductions < 64 && reduction_stride < 32) {
    U totals[31];
    for (int i = 0; i < 31; i++) {
      totals[i] = Op::init;
    }

    short stride = reduction_stride;
    short size = reduction_size;
    short blocks = stride / N_READS;
    short extra = stride - blocks * N_READS;

    size_t out_idx = tid.x + tsize.y * size_t(tid.y);
    in += elem_to_loc(out_idx, shape, strides, ndim);

    for (uint r = 0; r < non_col_reductions; r++) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);

      for (short i = 0; i < size; i++) {
        for (short j = 0; j < blocks; j++) {
          for (short k = 0; k < N_READS; k++) {
            totals[j * N_READS + k] =
                op(totals[j * N_READS + k],
                   static_cast<U>(row[i * stride + j * N_READS + k]));
          }
        }
        for (short k = 0; k < extra; k++) {
          totals[blocks * N_READS + k] =
              op(totals[blocks * N_READS + k],
                 static_cast<U>(row[i * stride + blocks * N_READS + k]));
        }
      }

      loop.next(reduce_shape, reduce_strides);
    }
    out += out_idx * reduction_stride;
    for (short j = 0; j < stride; j++) {
      out[j] = totals[j];
    }
  }

  // Case 2: Long row small column
  else if (reduction_size * non_col_reductions < 32) {
    U totals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      totals[i] = Op::init;
    }

    short size = reduction_size;
    size_t offset = size_t(tid.x) * N_READS;
    bool safe = offset + N_READS <= reduction_stride;
    short extra = reduction_stride - offset;

    size_t out_idx = tid.y + tsize.z * size_t(tid.z);
    in += elem_to_loc(out_idx, shape, strides, ndim) + offset;

    for (uint r = 0; r < non_col_reductions; r++) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);

      if (safe) {
        for (short i = 0; i < size; i++) {
          for (short j = 0; j < N_READS; j++) {
            totals[j] =
                op(static_cast<U>(row[i * reduction_stride + j]), totals[j]);
          }
        }
      } else {
        for (short i = 0; i < size; i++) {
          for (short j = 0; j < extra; j++) {
            totals[j] =
                op(static_cast<U>(row[i * reduction_stride + j]), totals[j]);
          }
        }
      }

      loop.next(reduce_shape, reduce_strides);
    }
    out += out_idx * reduction_stride + offset;
    if (safe) {
      for (short i = 0; i < N_READS; i++) {
        out[i] = totals[i];
      }
    } else {
      for (short i = 0; i < extra; i++) {
        out[i] = totals[i];
      }
    }
  }

  // Case 3: Long row medium column
  else {
    threadgroup U shared_vals[1024];
    U totals[N_READS];
    for (int i = 0; i < N_READS; i++) {
      totals[i] = Op::init;
    }

    short stride = reduction_stride;
    short lid = simd_group_id * simd_size + simd_lane_id;
    short2 tile((stride + N_READS - 1) / N_READS, 32);
    short2 offset((lid % tile.x) * N_READS, lid / tile.x);
    short sm_stride = tile.x * N_READS;
    bool safe = offset.x + N_READS <= stride;

    size_t out_idx = gid.y + gsize.y * size_t(gid.z);
    in += elem_to_loc(out_idx, shape, strides, ndim) + offset.x;

    // Read cooperatively and contiguously and aggregate the partial results.
    size_t total = non_col_reductions * reduction_size;
    loop.next(offset.y, reduce_shape, reduce_strides);
    for (size_t r = offset.y; r < total; r += simd_size) {
      row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);

      if (safe) {
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(static_cast<U>(row[i]), totals[i]);
        }
      } else {
        U vals[N_READS];
        for (int i = 0; i < N_READS; i++) {
          vals[i] = (offset.x + i < stride) ? static_cast<U>(row[i]) : op.init;
        }
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(vals[i], totals[i]);
        }
      }

      loop.next(simd_size, reduce_shape, reduce_strides);
    }

    // Each thread holds N_READS partial results but the simdgroups are not
    // aligned to do the reduction across the simdgroup so we write our results
    // in the shared memory and read them back according to the simdgroup.
    for (int i = 0; i < N_READS; i++) {
      shared_vals[offset.y * sm_stride + offset.x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      totals[i] = op.simd_reduce(
          shared_vals[simd_lane_id * sm_stride + simd_group_id * N_READS + i]);
    }

    // Write the output.
    if (simd_lane_id == 0) {
      short column = simd_group_id * N_READS;
      out += out_idx * reduction_stride + column;
      if (column + N_READS <= stride) {
        for (int i = 0; i < N_READS; i++) {
          out[i] = totals[i];
        }
      } else {
        for (int i = 0; column + i < stride; i++) {
          out[i] = totals[i];
        }
      }
    }
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
template <typename T, typename U, typename Op, int NDIMS, int BM, int BN>
[[kernel]] void col_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    const constant size_t& non_col_reductions [[buffer(10)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  constexpr int n_simdgroups = 4;
  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short n_read_blocks = BN / n_reads;

  threadgroup U shared_vals[BN * BM];
  U totals[n_reads];
  looped_elem_to_loc<NDIMS> loop;
  const device T* row;

  for (int i = 0; i < n_reads; i++) {
    totals[i] = Op::init;
  }

  short lid = simd_group_id * simd_size + simd_lane_id;
  short2 offset((lid % n_read_blocks) * n_reads, lid / n_read_blocks);
  size_t column = BN * gid.x + offset.x;
  bool safe = column + n_reads <= reduction_stride;

  size_t out_idx = gid.y + gsize.y * size_t(gid.z);
  size_t in_idx = elem_to_loc(out_idx, shape, strides, ndim);
  in += in_idx + column;

  size_t total = non_col_reductions * reduction_size;
  loop.next(offset.y, reduce_shape, reduce_strides);
  for (size_t r = offset.y; r < total; r += BM) {
    row = in + loop.location(r, reduce_shape, reduce_strides, reduce_ndim);

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
      size_t out_column = BN * gid.x + out_offset.x;
      out += out_idx * reduction_stride + out_column;
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
      out += out_idx * reduction_stride + column;
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
