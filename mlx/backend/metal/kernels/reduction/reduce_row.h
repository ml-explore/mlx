// Copyright © 2023-2024 Apple Inc.

// Row reduction utilities
// - `per_thread_row_reduce` collaborative partial reduction in the threadgroup
// - `threadgroup_reduce` collaborative reduction in the threadgroup such that
//   lid.x == 0 holds the reduced value
// - `thread_reduce` simple loop and reduce the row

/**
 * The thread group collaboratively reduces across the rows with bounds
 * checking. In the end each thread holds a part of the reduction.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* inputs[N_WRITES],
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  Op op;

  // Set up the accumulator registers
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = Op::init;
  }

  // Loop over the reduction size within thread group
  for (int i = 0; i < blocks; i++) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }

      inputs[j] += lsize_x * N_READS;
    }
  }

  // Separate case for the last set as we close the reduction size
  int index = lid_x * N_READS;
  if (index + N_READS <= extra) {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  } else {
    for (int j = 0; j < N_WRITES; j++) {
      for (int i = 0; index + i < extra; i++) {
        totals[j] = op(static_cast<U>(inputs[j][i]), totals[j]);
      }
    }
  }
}

/**
 * Consecutive rows in a contiguous array.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const constant size_t& reduction_size,
    int blocks,
    int extra,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Consecutive rows in an arbitrarily ordered array.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void per_thread_row_reduce(
    thread U totals[N_WRITES],
    const device T* in,
    const int64_t row_idx,
    int blocks,
    int extra,
    const constant int* shape,
    const constant int64_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  in += lid_x * N_READS;
  for (int i = 0; i < N_READS; i++) {
    inputs[i] = in + elem_to_loc(row_idx + i, shape, strides, ndim);
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, blocks, extra, lsize_x, lid_x);
}

/**
 * Reduce within the threadgroup.
 */
template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
METAL_FUNC void threadgroup_reduce(
    thread U totals[N_WRITES],
    threadgroup U* shared_vals,
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  // Simdgroup first
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = op.simd_reduce(totals[i]);
  }

  // Across simdgroups
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      for (int i = 0; i < N_WRITES; i++) {
        shared_vals[simd_group_id * N_WRITES + i] = totals[i];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    U values[N_WRITES];
    for (int i = 0; i < N_WRITES; i++) {
      values[i] = (lid.x < simd_per_group) ? shared_vals[lid.x * N_WRITES + i]
                                           : op.init;
    }

    for (int i = 0; i < N_WRITES; i++) {
      totals[i] = op.simd_reduce(values[i]);
    }
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC void
thread_reduce(thread U& total, const device T* row, int blocks, int extra) {
  Op op;
  for (int i = 0; i < blocks; i++) {
    U vals[N_READS];
    for (int j = 0; j < N_READS; j++) {
      vals[j] = row[j];
    }
    for (int j = 0; j < N_READS; j++) {
      total = op(vals[j], total);
    }
    row += N_READS;
  }
  for (int i = 0; i < extra; i++) {
    total = op(*row++, total);
  }
}

// Reduction kernels
// - `row_reduce_small` depending on the non-row reductions and row size it
//   either just loops over everything or a simd collaboratively reduces the
//   non_row reductions. In the first case one thread is responsible for one
//   output on the 2nd one simd is responsible for one output.
// - `row_reduce_simple` simple contiguous row reduction
// - `row_reduce_looped` simply loop and reduce each row for each non-row
//   reduction. One threadgroup is responsible for one output.

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 tid [[thread_position_in_grid]],
    uint3 tsize [[threads_per_grid]]) {
  Op op;

  U total_val = Op::init;
  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);

  // Precompute some row reduction numbers
  const device T* row;
  int blocks = IdxT(row_size) / N_READS;
  int extra = IdxT(row_size) % N_READS;

  if ((non_row_reductions < 32 && row_size <= 8) || non_row_reductions <= 8) {
    // Simple loop over non_row_reductions and reduce the row in the thread.
    IdxT out_idx = tid.x + tsize.y * IdxT(tid.y);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    for (uint r = 0; r < non_row_reductions; r++) {
      row = in + loop.location();
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(reduce_shape, reduce_strides);
    }

    out[out_idx] = total_val;
  } else {
    // Collaboratively reduce over non_row_reductions in the simdgroup. Each
    // thread reduces every 32nd row and then a simple simd reduce.
    IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);
    in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim);

    loop.next(simd_lane_id, reduce_shape, reduce_strides);

    for (uint r = simd_lane_id; r < non_row_reductions; r += simd_size) {
      row = in + loop.location();
      thread_reduce<T, U, Op, N_READS>(total_val, row, blocks, extra);
      loop.next(simd_size, reduce_shape, reduce_strides);
    }

    total_val = op.simd_reduce(total_val);

    if (simd_lane_id == 0) {
      out[out_idx] = total_val;
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
[[kernel]] void row_reduce_simple(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant int64_t& out_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  threadgroup U shared_vals[simd_size * N_WRITES];
  U totals[N_WRITES];

  // Move to the row
  IdxT out_idx = N_WRITES * (gid.y + gsize.y * IdxT(gid.z));
  if (out_idx + N_WRITES > out_size) {
    out_idx = out_size - N_WRITES;
  }
  in += out_idx * IdxT(reduction_size);
  out += out_idx;

  // Each thread reduces across the row
  int blocks = IdxT(reduction_size) / (lsize.x * N_READS);
  int extra = reduction_size - blocks * (lsize.x * N_READS);
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, in, reduction_size, blocks, extra, lsize.x, lid.x);

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    for (int i = 0; i < N_WRITES; i++) {
      out[i] = totals[i];
    }
  }
}

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT,
    int NDIMS,
    int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int64_t& row_size [[buffer(2)]],
    const constant int64_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant int64_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant int64_t* reduce_strides [[buffer(8)]],
    const constant int& reduce_ndim [[buffer(9)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];
  U total = Op::init;

  IdxT out_idx = gid.y + gsize.y * IdxT(gid.z);

  // lid.x * N_READS breaks the per_thread_row_reduce interface a bit. Maybe it
  // needs a small refactor.
  in += elem_to_loc<IdxT>(out_idx, shape, strides, ndim) + lid.x * N_READS;

  LoopedElemToLoc<NDIMS, IdxT, (NDIMS > 2)> loop(reduce_ndim);
  const device T* row;
  int blocks = IdxT(row_size) / (lsize.x * N_READS);
  int extra = row_size - blocks * (lsize.x * N_READS);

  for (IdxT i = 0; i < non_row_reductions; i++) {
    row = in + loop.location();

    // Each thread reduces across the row
    U row_total;
    per_thread_row_reduce<T, U, Op, N_READS, 1>(
        &row_total, &row, blocks, extra, lsize.x, lid.x);

    // Aggregate across rows
    total = op(total, row_total);

    loop.next(reduce_shape, reduce_strides);
  }

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, 1>(
      &total, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    out[out_idx] = total;
  }
}
