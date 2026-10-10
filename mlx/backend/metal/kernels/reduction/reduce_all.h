// Copyright © 2023-2024 Apple Inc.

template <
    typename T,
    typename U,
    typename Op,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant size_t& in_size [[buffer(2)]],
    const constant size_t& row_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U shared_vals[simd_size];

  U total = Op::init;
  IdxT start_idx = gid.y * IdxT(row_size);
  IdxT actual_row =
      (start_idx + row_size <= in_size) ? row_size : in_size - start_idx;
  IdxT blocks = actual_row / (lsize.x * N_READS);
  int extra = actual_row - blocks * (lsize.x * N_READS);
  extra -= lid.x * N_READS;
  start_idx += lid.x * N_READS;
  in += start_idx;

  if (extra >= N_READS) {
    blocks++;
    extra = 0;
  }

  for (IdxT b = 0; b < blocks; b++) {
    for (int i = 0; i < N_READS; i++) {
      total = op(cast_to<U>(in[i]), total);
    }
    in += lsize.x * N_READS;
  }
  if (extra > 0) {
    for (int i = 0; i < extra; i++) {
      total = op(cast_to<U>(in[i]), total);
    }
  }

  // Reduction within simd group
  total = op.simd_reduce(total);
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      shared_vals[simd_group_id] = total;
    }

    // Reduction within thread group
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = lid.x < simd_per_group ? shared_vals[lid.x] : op.init;
    total = op.simd_reduce(total);
  }

  if (lid.x == 0) {
    out[gid.y] = total;
  }
}

template <
    typename T,
    typename U,
    typename IdxT = int64_t,
    int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce_min_max(
    const device T* in [[buffer(0)]],
    device U* out1 [[buffer(1)]],
    device U* out2 [[buffer(4)]],
    const constant size_t& in_size [[buffer(2)]],
    const constant size_t& row_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Min<U> min_op;
  Max<U> max_op;

  threadgroup U shared_min[simd_size];
  threadgroup U shared_max[simd_size];

  U min_total = Min<U>::init;
  U max_total = Max<U>::init;
  IdxT start_idx = gid.y * IdxT(row_size);
  IdxT actual_row =
      (start_idx + row_size <= in_size) ? row_size : in_size - start_idx;
  IdxT blocks = actual_row / (lsize.x * N_READS);
  int extra = actual_row - blocks * (lsize.x * N_READS);
  extra -= lid.x * N_READS;
  start_idx += lid.x * N_READS;
  in += start_idx;

  if (extra >= N_READS) {
    blocks++;
    extra = 0;
  }
  U val;
  for (IdxT b = 0; b < blocks; b++) {
    for (int i = 0; i < N_READS; i++) {
      val = cast_to<U>(in[i]);
      min_total = min_op(val, min_total);
      max_total = max_op(val, max_total);
    }
    in += lsize.x * N_READS;
  }
  if (extra > 0) {
    for (int i = 0; i < extra; i++) {
      val = cast_to<U>(in[i]);
      min_total = min_op(val, min_total);
      max_total = max_op(val, max_total);
    }
  }

  // Reduction within simd group
  min_total = min_op.simd_reduce(min_total);
  max_total = max_op.simd_reduce(max_total);
  if (simd_per_group > 1) {
    if (simd_lane_id == 0) {
      shared_min[simd_group_id] = min_total;
      shared_max[simd_group_id] = max_total;
    }

    // Reduction within thread group
    threadgroup_barrier(mem_flags::mem_threadgroup);
    min_total = lid.x < simd_per_group ? shared_min[lid.x] : min_op.init;
    min_total = min_op.simd_reduce(min_total);
    max_total = lid.x < simd_per_group ? shared_max[lid.x] : max_op.init;
    max_total = max_op.simd_reduce(max_total);
  }

  if (lid.x == 0) {
    out1[gid.y] = min_total;
    out2[gid.y] = max_total;
  }
}
