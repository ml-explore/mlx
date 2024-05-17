// Copyright © 2023-2024 Apple Inc.

///////////////////////////////////////////////////////////////////////////////
// All reduce helper
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

///////////////////////////////////////////////////////////////////////////////
// All reduce kernel
///////////////////////////////////////////////////////////////////////////////

// NB: This kernel assumes threads_per_threadgroup is at most
// 1024. This way with a simd_size of 32, we are guaranteed to
// complete the reduction in two steps of simd-level reductions.
template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T* in [[buffer(0)]],
    device mlx_atomic<U>* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  Op op;
  threadgroup U local_vals[simd_size];

  U total_val =
      per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);

  // Reduction within simd group
  total_val = op.simd_reduce(total_val);
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }

  // Reduction within thread group
  threadgroup_barrier(mem_flags::mem_threadgroup);
  total_val = lid < simd_per_group ? local_vals[lid] : op.init;
  total_val = op.simd_reduce(total_val);

  // Reduction across threadgroups
  if (lid == 0) {
    op.atomic_update(out, total_val);
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void all_reduce_no_atomics(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint thread_group_id [[threadgroup_position_in_grid]]) {
  Op op;
  threadgroup U local_vals[simd_size];

  U total_val =
      per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);

  // Reduction within simd group (simd_add isn't supported for uint64/int64
  // types)
  for (uint16_t lane_offset = simd_size / 2; lane_offset > 0;
       lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }
  // Write simd group reduction results to local memory
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction of simdgroup reduction results within threadgroup.
  total_val = lid < simd_per_group ? local_vals[lid] : op.init;
  for (uint16_t lane_offset = simd_size / 2; lane_offset > 0;
       lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }

  // Reduction across threadgroups
  if (lid == 0) {
    out[thread_group_id] = total_val;
  }
}
