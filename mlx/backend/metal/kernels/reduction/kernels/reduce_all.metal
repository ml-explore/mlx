// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/reduction/utils.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_all.h"
#include "mlx/backend/metal/kernels/reduction/reduce_inst.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// All reduce
///////////////////////////////////////////////////////////////////////////////


// NB: This kernel assumes threads_per_threadgroup is at most
// 1024. This way with a simd_size of 32, we are guaranteed to
// complete the reduction in two steps of simd-level reductions.
template <typename T, typename U, typename Op, int N_READS=REDUCE_N_READS>
[[kernel]] void all_reduce(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const device size_t& in_size [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint grid_size [[threads_per_grid]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  Op op;
  threadgroup U local_vals[simd_size];

  U total_val = per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);

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

template <typename T, typename U, typename Op, int N_READS=REDUCE_N_READS>
[[kernel]] void all_reduce_no_atomics(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
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

  U total_val = per_thread_all_reduce<T, U, Op, N_READS>(in, in_size, gid, grid_size);

  // Reduction within simd group (simd_add isn't supported for uint64/int64 types)
  for (uint16_t lane_offset = simd_size/2; lane_offset > 0; lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }
  // Write simd group reduction results to local memory
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction of simdgroup reduction results within threadgroup.
  total_val = lid < simd_per_group ? local_vals[lid] : op.init;
  for (uint16_t lane_offset = simd_size/2; lane_offset > 0; lane_offset /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, lane_offset));
  }

  // Reduction across threadgroups
  if (lid == 0) {
    out[thread_group_id] = total_val;
  }
}

#define instantiate_all_reduce(name, itype, otype, op) \
  template [[host_name("all_reduce_" #name)]] \
  [[kernel]] void all_reduce<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device mlx_atomic<otype> *out [[buffer(1)]], \
      const device size_t& in_size [[buffer(2)]], \
      uint gid [[thread_position_in_grid]], \
      uint lid [[thread_position_in_threadgroup]], \
      uint grid_size [[threads_per_grid]], \
      uint simd_per_group [[simdgroups_per_threadgroup]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_all_reduce_no_atomics(name, itype, otype, op) \
  template [[host_name("all_reduce_no_atomics_" #name)]] \
  [[kernel]] void all_reduce_no_atomics<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device otype *out [[buffer(1)]], \
      const device size_t& in_size [[buffer(2)]], \
      uint gid [[thread_position_in_grid]], \
      uint lid [[thread_position_in_threadgroup]], \
      uint grid_size [[threads_per_grid]], \
      uint simd_per_group [[simdgroups_per_threadgroup]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]], \
      uint thread_group_id [[threadgroup_position_in_grid]]);

///////////////////////////////////////////////////////////////////////////////
// Instantiations
///////////////////////////////////////////////////////////////////////////////


#define instantiate_same_all_reduce_helper(name, tname, type, op) \
  instantiate_all_reduce(name ##tname, type, type, op<type>)

#define instantiate_same_all_reduce_na_helper(name, tname, type, op) \
  instantiate_all_reduce_no_atomics(name ##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_all_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_all_reduce_na_helper, instantiate_reduce_helper_64b)

instantiate_reduce_from_types(instantiate_all_reduce, and, bool, And)
instantiate_reduce_from_types(instantiate_all_reduce, or, bool, Or)

// special case bool with larger output type
instantiate_all_reduce(sumbool_, bool, uint32_t, Sum<uint32_t>)