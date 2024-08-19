// Copyright © 2023-2024 Apple Inc.

template <
    typename T,
    typename U,
    typename Op,
    int NDIMS = 0,
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

  // Case 1:
  // reduction_stride is small, reduction_size is small and non_col_reductions
  // is small. Each thread computes reduction_stride outputs.
  if (reduction_size * non_col_reductions < 64) {
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
      const device T* in_row;
      if constexpr (loop.dynamic_ndim) {
        in_row = in + elem_to_loc(r, reduce_shape, reduce_strides, reduce_ndim);
      } else {
        in_row = in + loop.offset;
      }

      for (short i = 0; i < size; i++) {
        for (short j = 0; j < blocks; j++) {
          for (short k = 0; k < N_READS; k++) {
            totals[j * N_READS + k] =
                op(totals[j * N_READS + k],
                   static_cast<U>(in_row[i * stride + j * N_READS + k]));
          }
        }
        for (short k = 0; k < extra; k++) {
          totals[blocks * N_READS + k] =
              op(totals[blocks * N_READS + k],
                 static_cast<U>(in_row[i * stride + blocks * N_READS + k]));
        }
      }

      loop.next(reduce_shape, reduce_strides);
    }
    out += out_idx * reduction_stride;
    for (short j = 0; j < stride; j++) {
      out[j] = totals[j];
    }
  }

  // Case 2:
  // Reduction stride is small but everything else can be big. We loop bot
  // across reduction size and non_col_reductions.
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

    size_t total = non_col_reductions * reduction_size;
    loop.next(offset.y, reduce_shape, reduce_strides);
    for (size_t r = offset.y; r < total; r += simd_size) {
      const device T* in_row;
      if constexpr (loop.dynamic_ndim) {
        in_row = in + elem_to_loc(r, reduce_shape, reduce_strides, reduce_ndim);
      } else {
        in_row = in + loop.offset;
      }

      if (safe) {
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(static_cast<U>(in_row[i]), totals[i]);
        }
      } else {
        U vals[N_READS];
        for (int i = 0; i < N_READS; i++) {
          vals[i] =
              (offset.x + i < stride) ? static_cast<U>(in_row[i]) : op.init;
        }
        for (int i = 0; i < N_READS; i++) {
          totals[i] = op(vals[i], totals[i]);
        }
      }

      loop.next(simd_size, reduce_shape, reduce_strides);
    }

    for (int i = 0; i < N_READS; i++) {
      shared_vals[offset.y * sm_stride + offset.x + i] = totals[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = 0; i < N_READS; i++) {
      totals[i] = op.simd_reduce(
          shared_vals[simd_lane_id * sm_stride + simd_group_id * N_READS + i]);
    }

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
 *  2. Load a tile BM, BN in shared memory.
 *  3. Add the values from shared memory to the current running totals.
 *     Neighboring threads access different rows (transposed acces).
 *  4. Move ahead to the next tile until the M axis is exhausted.
 *  5. Move ahead to the next non column reduction
 *  6. Simd reduce the running totals
 *  7. Write them to the output
 *
 * The kernel becomes verbose because we support all kinds of OOB checks. For
 * instance if we choose that reduction_stride must be larger than BN then we
 * can get rid of half the kernel.
 */
template <
    typename T,
    typename U,
    typename Op,
    int NDIMS = 0,
    int BM = 32,
    int BN_32bit = 32>
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
  constexpr int BN = (BN_32bit * sizeof(T) <= 128) ? BN_32bit : BN_32bit / 2;
  constexpr int BN_padded = (BN + 16 / sizeof(T));
  constexpr int n_outputs = BN / n_simdgroups;
  threadgroup T shared_vals[BM * BN_padded];
  thread U totals[n_outputs];
  threadgroup T* shared_vals_local =
      shared_vals + simd_lane_id * BN_padded + simd_group_id * n_outputs;

  for (int i = 0; i < n_outputs; i++) {
    totals[i] = Op::init;
  }

  constexpr short tgp_size = n_simdgroups * simd_size;
  constexpr short n_reads = (BM * BN) / tgp_size;
  constexpr short TCOLS = BN / n_reads;
  using loader_t = mlx::steel::BlockLoader<
      T,
      /*         BROWS= */ BM,
      /*         BCOLS= */ BN,
      /*        dst_ld= */ BN_padded,
      /* reduction_dim= */ 0,
      /*      tgp_size= */ tgp_size,
      /*     alignment= */ 1,
      /*       n_reads= */ n_reads,
      /*         TCOLS= */ TCOLS,
      /*         TROWS= */ tgp_size / TCOLS,
      /*   initializer= */ Op>;

  // If the column is OOB then move it back so that we always have BN columns
  // to read.
  size_t column = BN * gid.x;
  if (column + BN > reduction_stride && BN < reduction_stride) {
    column = reduction_stride - BN;
  }

  size_t out_idx = gid.y + gsize.y * size_t(gid.z);
  size_t in_idx = elem_to_loc(out_idx, shape, strides, ndim);
  const int n_blocks = reduction_size / BM;
  bool extra = (reduction_size % BM) != 0;

  out += out_idx * reduction_stride + column + simd_group_id * n_outputs;
  in += in_idx + column;

  loader_t loader(
      in, reduction_stride, shared_vals, simd_group_id, simd_lane_id);
  looped_elem_to_loc<NDIMS> loop;

  if (reduction_stride < BN) {
    short2 tile(reduction_stride, BM);
    for (size_t r = 0; r < non_col_reductions; r++) {
      if constexpr (loop.dynamic_ndim) {
        loader.reset(
            in + elem_to_loc(r, reduce_shape, reduce_strides, reduce_ndim));
      } else {
        loader.reset(in + loop.offset);
      }

      for (int i = 0; i < n_blocks; i++) {
        loader.load_safe(tile);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Partial reduction of the whole block
        for (int i = 0; i < n_outputs; i++) {
          U val = shared_vals_local[i];
          totals[i] = op(val, totals[i]);
        }

        // Load the next block
        loader.next();
      }
      if (extra) {
        // Load the last block with bounds checking
        loader.load_safe(short2(tile.x, reduction_size - n_blocks * BM));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Partial reduction of the whole block
        for (int i = 0; i < n_outputs; i++) {
          U val = shared_vals_local[i];
          totals[i] = op(val, totals[i]);
        }
      }

      loop.next(reduce_shape, reduce_strides);
    }
  } else {
    for (size_t r = 0; r < non_col_reductions; r++) {
      if constexpr (loop.dynamic_ndim) {
        loader.reset(
            in + elem_to_loc(r, reduce_shape, reduce_strides, reduce_ndim));
      } else {
        loader.reset(in + loop.offset);
      }

      for (int i = 0; i < n_blocks; i++) {
        loader.load_unsafe();
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Partial reduction of the whole block
        for (int i = 0; i < n_outputs; i++) {
          U val = shared_vals_local[i];
          totals[i] = op(val, totals[i]);
        }

        // Load the next block
        loader.next();
      }
      if (extra) {
        // Load the last block with bounds checking
        loader.load_safe(short2(BN, reduction_size - n_blocks * BM));
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Partial reduction of the whole block
        for (int i = 0; i < n_outputs; i++) {
          U val = shared_vals_local[i];
          totals[i] = op(val, totals[i]);
        }
      }

      loop.next(reduce_shape, reduce_strides);
    }
  }

  // Reduce across simdgroups
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < n_outputs; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }

  // Write output
  if (simd_lane_id == 0) {
    if (column + (simd_group_id + 1) * n_outputs <= reduction_stride) {
      for (int i = 0; i < n_outputs; i++) {
        out[i] = totals[i];
      }
    } else {
      for (uint i = 0, j = column + simd_group_id * n_outputs;
           j < reduction_stride;
           i++, j++) {
        out[i] = totals[i];
      }
    }
  }
}
