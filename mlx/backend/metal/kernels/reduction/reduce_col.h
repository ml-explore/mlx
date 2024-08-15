// Copyright © 2023-2024 Apple Inc.

///////////////////////////////////////////////////////////////////////////////
// Small column reduce kernel
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op>
[[kernel]] void col_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    const constant size_t& non_col_reductions [[buffer(8)]],
    const constant int* non_col_shapes [[buffer(9)]],
    const constant size_t* non_col_strides [[buffer(10)]],
    const constant int& non_col_ndim [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  // Appease the compiler
  (void)out_size;

  Op op;
  U total_val = Op::init;

  auto out_idx = tid;

  in += elem_to_loc(
      out_idx,
      shape + non_col_ndim,
      strides + non_col_ndim,
      ndim - non_col_ndim);

  for (uint i = 0; i < non_col_reductions; i++) {
    size_t in_idx =
        elem_to_loc(i, non_col_shapes, non_col_strides, non_col_ndim);

    for (uint j = 0; j < reduction_size; j++, in_idx += reduction_stride) {
      U val = static_cast<U>(in[in_idx]);
      total_val = op(total_val, val);
    }
  }

  out[out_idx] = total_val;
}

///////////////////////////////////////////////////////////////////////////////
// Column reduce helper
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U _contiguous_strided_reduce(
    const device T* in,
    threadgroup U* local_data,
    size_t in_idx,
    size_t reduction_size,
    size_t reduction_stride,
    uint2 tid,
    uint2 lid,
    uint2 lsize) {
  Op op;
  U total_val = Op::init;

  size_t base_offset = (size_t(tid.y) * lsize.y + lid.y) * N_READS;

  in += in_idx;
  in += base_offset * reduction_stride;

  if (base_offset + N_READS <= reduction_size) {
    for (int r = 0; r < N_READS; r++) {
      total_val = op(static_cast<U>(total_val), *in);
      in += reduction_stride;
    }
  } else {
    int remaining = reduction_size - base_offset;
    for (int r = 0; r < remaining; r++) {
      total_val = op(static_cast<U>(total_val), *in);
      in += reduction_stride;
    }
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

///////////////////////////////////////////////////////////////////////////////
// Column reduce kernel
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    typename Op,
    int NDIMS = 0,
    int BM = 32,
    int BN = 64>
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

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_general_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup U* local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]]) {
  auto out_idx = size_t(tid.x) * lsize.x + lid.x;
  auto in_idx = elem_to_loc(out_idx + tid.z * out_size, shape, strides, ndim);

  if (out_idx < out_size) {
    U val = _contiguous_strided_reduce<T, U, Op, N_READS>(
        in,
        local_data,
        in_idx,
        reduction_size,
        reduction_stride,
        tid.xy,
        lid.xy,
        lsize.xy);

    // Write out reduction results generated by threadgroups working on specific
    // output element, contiguously.
    if (lid.y == 0) {
      uint tgsize_y = ceildiv(gsize.y, lsize.y);
      uint tgsize_z = ceildiv(gsize.z, lsize.z);
      out[tgsize_y * tgsize_z * gid.x + tgsize_y * tid.z + tid.y] = val;
    }
  }
}
