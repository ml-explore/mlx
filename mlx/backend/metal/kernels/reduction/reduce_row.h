// Copyright © 2023-2024 Apple Inc.

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

/**
 * Reduce across a row with bounds checking but each thread in the threadgroup
 * holds a part of the reduction.
 */
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U per_thread_row_reduce(
    const device T* in,
    const constant int& reduction_size,
    uint lsize_x,
    uint lid_x) {
  Op op;

  in += lid_x * N_READS;

  // The reduction is accumulated here
  U total_val = Op::init;

  // Loop over the reduction size within thread group
  int r = 0;
  for (; r < ceildiv(reduction_size, N_READS * lsize_x) - 1; r++) {
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
  int reduction_index = (lid_x + lsize_x * r) * N_READS;
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

/**
 * Reduce within the threadgroup.
 */
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U threadgroup_reduce(
    U total_val,
    threadgroup U* shared_vals,
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;

  // Simdgroup first
  total_val = op.simd_reduce(total_val);

  // Across simdgroups
  if (simd_lane_id == 0) {
    shared_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_per_group > 1) {
    total_val = lid.x < simd_per_group ? shared_vals[lid.x] : op.init;
    total_val = op.simd_reduce(total_val);
  }

  return total_val;
}

/**
 * Reduce across a row with bounds checking but each thread in the threadgroup
 * holds a part of the reduction.
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
  Op op;

  // Set up the input pointers
  const device T* inputs[N_WRITES];
  inputs[0] = in + lid_x * N_READS;
  for (int i = 1; i < N_READS; i++) {
    inputs[i] = inputs[i - 1] + reduction_size;
  }

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
  if (simd_lane_id == 0) {
    for (int i = 0; i < N_WRITES; i++) {
      shared_vals[simd_group_id * N_WRITES + i] = totals[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_per_group > 1) {
    if (lid.x < simd_per_group) {
      for (int i = 0; i < N_WRITES; i++) {
        totals[i] = op.simd_reduce(shared_vals[lid.x * N_WRITES + i]);
      }
    } else {
      for (int i = 0; i < N_WRITES; i++) {
        totals[i] = op.simd_reduce(op.init);
      }
    }
  }
}

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
[[kernel]] void row_reduce_general(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const constant int& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  (void)non_row_reductions;

  Op op;
  threadgroup U local_vals[simd_size];

  size_t idx = size_t(tid.y) * out_size + tid.x;
  size_t extra_offset = elem_to_loc(idx, shape, strides, ndim);
  in += extra_offset;

  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(
      in, reduction_size, lsize.x, lid.x);

  total_val = op.simd_reduce(total_val);

  // Prepare next level
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction within thread group
  //    Only needed if multiple simd groups
  if (reduction_size > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    total_val = op.simd_reduce(total_val);
  }
  // Update output
  if (lid.x == 0) {
    op.atomic_update(out, total_val, tid.x);
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void row_reduce_general_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  (void)non_row_reductions;

  Op op;
  threadgroup U local_vals[simd_size];

  size_t idx = size_t(tid.y) * out_size + tid.x;
  size_t extra_offset = elem_to_loc(idx, shape, strides, ndim);
  in += extra_offset;

  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(
      in, reduction_size, lsize.x, lid.x);

  // Reduction within simd group - simd_add isn't supported for int64 types
  for (uint16_t i = simd_size / 2; i > 0; i /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, i));
  }

  // Prepare next level
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction within thread group
  // Only needed if thread group has multiple simd groups
  if (ceildiv(reduction_size, N_READS) > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    for (uint16_t i = simd_size / 2; i > 0; i /= 2) {
      total_val = op(total_val, simd_shuffle_down(total_val, i));
    }
  }
  // Write row reduce output for threadgroup with 1st thread in thread group
  if (lid.x == 0) {
    out[(ceildiv(gsize.y, lsize.y) * tid.x) + tid.y] = total_val;
  }
}
