// Copyright © 2023-2024 Apple Inc.

// Row reduction utilities
// - `per_thread_row_reduce` collaborative partial reduction in the threadgroup
// - `threadgroup_reduce` collaborative reduction in the threadgroup such that
//   lid.x == 0 holds the reduced value

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
    const constant int& reduction_size,
    uint lsize_x,
    uint lid_x) {
  Op op;

  // Set up the accumulator registers
  for (int i = 0; i < N_WRITES; i++) {
    totals[i] = Op::init;
  }

  // Loop over the reduction size within thread group
  int r = 0;
  for (; r < ceildiv(reduction_size, N_READS * lsize_x) - 1; r++) {
    for (int j = 0; j < N_WRITES; j++) {
      T vals[N_READS];
      for (int i = 0; i < N_READS; i++) {
        vals[i] = inputs[j][i];
      }
      for (int i = 0; i < N_READS; i++) {
        totals[j] = op(static_cast<U>(vals[i]), totals[j]);
      }

      inputs[j] += lsize_x * N_READS;
    }
  }

  // Separate case for the last set as we close the reduction size
  int reduction_index = (lid_x + lsize_x * r) * N_READS;
  if (reduction_index < reduction_size) {
    int max_reads = reduction_size - reduction_index;

    for (int j = 0; j < N_WRITES; j++) {
      T vals[N_READS];
      for (int i = 0; i < N_READS; i++) {
        int idx = min(i, max_reads - 1);
        vals[i] = inputs[j][idx];
      }
      for (int i = 0; i < N_READS; i++) {
        T val = i < max_reads ? vals[i] : Op::init;
        totals[j] = op(static_cast<U>(val), totals[j]);
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
    const constant int& reduction_size,
    uint lsize_x,
    uint lid_x) {
  // Set up the input pointers
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }

  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, inputs, reduction_size, lsize_x, lid_x);
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
    const size_t row_idx,
    const constant int& reduction_size,
    const constant int* shape,
    const constant size_t* strides,
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
      totals, inputs, reduction_size, lsize_x, lid_x);
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

///////////////////////////////////////////////////////////////////////////////
// Small row reductions
///////////////////////////////////////////////////////////////////////////////

// Each thread reduces for one output
template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_small(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint lid [[thread_position_in_grid]]) {
  Op op;

  uint out_idx = lid;

  if (out_idx >= out_size) {
    return;
  }

  U total_val = Op::init;

  for (short r = 0; r < short(non_row_reductions); r++) {
    uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
    const device T* in_row = in + in_idx;

    for (short i = 0; i < short(reduction_size); i++) {
      total_val = op(static_cast<U>(in_row[i]), total_val);
    }
  }

  out[out_idx] = total_val;
}

// Each simdgroup reduces for one output
template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_med(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[dispatch_simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  uint out_idx = simd_per_group * tid + simd_group_id;

  if (out_idx >= out_size) {
    return;
  }

  U total_val = Op::init;

  if (short(non_row_reductions) == 1) {
    uint in_idx = elem_to_loc(out_idx, shape, strides, ndim);
    const device T* in_row = in + in_idx;

    for (short i = simd_lane_id; i < short(reduction_size); i += 32) {
      total_val = op(static_cast<U>(in_row[i]), total_val);
    }
  }

  else if (short(non_row_reductions) >= 32) {
    for (short r = simd_lane_id; r < short(non_row_reductions); r += 32) {
      uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
      const device T* in_row = in + in_idx;

      for (short i = 0; i < short(reduction_size); i++) {
        total_val = op(static_cast<U>(in_row[i]), total_val);
      }
    }

  }

  else {
    const short n_reductions =
        short(reduction_size) * short(non_row_reductions);
    const short reductions_per_thread =
        (n_reductions + simd_size - 1) / simd_size;

    const short r_st = simd_lane_id / reductions_per_thread;
    const short r_ed = short(non_row_reductions);
    const short r_jump = simd_size / reductions_per_thread;

    const short i_st = simd_lane_id % reductions_per_thread;
    const short i_ed = short(reduction_size);
    const short i_jump = reductions_per_thread;

    if (r_st < r_jump) {
      for (short r = r_st; r < r_ed; r += r_jump) {
        uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
        const device T* in_row = in + in_idx;

        for (short i = i_st; i < i_ed; i += i_jump) {
          total_val = op(static_cast<U>(in_row[i]), total_val);
        }
      }
    }
  }

  total_val = op.simd_reduce(total_val);

  if (simd_lane_id == 0) {
    out[out_idx] = total_val;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Large row reductions
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    typename Op,
    int N_READS = REDUCE_N_READS,
    int N_WRITES = REDUCE_N_WRITES>
[[kernel]] void row_reduce_simple(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
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
  size_t out_idx = N_WRITES * (gid.y + gsize.y * size_t(gid.z));
  if (out_idx + N_WRITES > out_size) {
    out_idx = out_size - N_WRITES;
  }
  in += out_idx * reduction_size;
  out += out_idx;

  // Each thread reduces across the row
  per_thread_row_reduce<T, U, Op, N_READS, N_WRITES>(
      totals, in, reduction_size, lsize.x, lid.x);

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

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_looped(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int& row_size [[buffer(2)]],
    const constant size_t& non_row_reductions [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    const constant int* reduce_shape [[buffer(7)]],
    const constant size_t* reduce_strides [[buffer(8)]],
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

  size_t out_idx = gid.y + gsize.y * size_t(gid.z);

  in += elem_to_loc(out_idx, shape, strides, ndim);

  for (size_t i = 0; i < non_row_reductions; i++) {
    U row_total;

    // Each thread reduces across the row
    per_thread_row_reduce<T, U, Op, N_READS, 1>(
        &row_total,
        in,
        i,
        row_size,
        reduce_shape,
        reduce_strides,
        reduce_ndim,
        lsize.x,
        lid.x);

    // Aggregate across rows
    total = op(total, row_total);
  }

  // Reduce across the threadgroup
  threadgroup_reduce<T, U, Op, N_READS, 1>(
      &total, shared_vals, lid, simd_lane_id, simd_per_group, simd_group_id);

  // Write the output
  if (lid.x == 0) {
    out[out_idx] = total;
  }
}
