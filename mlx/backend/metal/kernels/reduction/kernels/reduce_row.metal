// Copyright © 2024 Apple Inc.

#include "mlx/backend/metal/kernels/reduction/utils.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_row.h"
#include "mlx/backend/metal/kernels/reduction/reduce_inst.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Row atomics
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS=REDUCE_N_READS>
[[kernel]] void row_reduce_general(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  Op op;
  threadgroup U local_vals[simd_size];

  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(in, reduction_size, out_size, shape, strides, ndim, lsize.x, lid.x, tid.xy);

  total_val = op.simd_reduce(total_val);
  
  // Prepare next level
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
      
  // Reduction within thread group
  //    Only needed if multiple simd groups
  if(reduction_size > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    total_val = op.simd_reduce(total_val);
  }
  // Update output
  if (lid.x == 0) {
    op.atomic_update(out, total_val, tid.x);
  }
}

template <typename T, typename U, typename Op, int N_READS=REDUCE_N_READS>
[[kernel]] void row_reduce_general_no_atomics(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  Op op;

  threadgroup U local_vals[simd_size];
  U total_val = per_thread_row_reduce<T, U, Op, N_READS>(in, reduction_size, out_size, shape, strides, ndim, lsize.x, lid.x, tid.xy);

  // Reduction within simd group - simd_add isn't supported for int64 types
  for (uint16_t i = simd_size/2; i > 0; i /= 2) {
    total_val = op(total_val, simd_shuffle_down(total_val, i));
  }

  // Prepare next level
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Reduction within thread group
  // Only needed if thread group has multiple simd groups
  if(ceildiv(reduction_size, N_READS) > simd_size) {
    total_val = lid.x < simd_per_group ? local_vals[lid.x] : op.init;
    for (uint16_t i = simd_size/2; i > 0; i /= 2) {
      total_val = op(total_val, simd_shuffle_down(total_val, i));
    }
  }
  // Write row reduce output for threadgroup with 1st thread in thread group
  if (lid.x == 0) {
    out[(ceildiv(gsize.y, lsize.y) * tid.x) + tid.y] = total_val;
  }
}

template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_small(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    uint lid [[thread_position_in_grid]]) {

  Op op;
  
  uint out_idx = lid;

  if(out_idx >= out_size) {
    return;
  }

  uint in_idx = elem_to_loc(out_idx, shape, strides, ndim);
  in += in_idx;

  U total_val = Op::init;
  
  for(short i = 0; i < short(reduction_size); i++) {
    total_val = op(static_cast<U>(in[i]), total_val);
  }

  out[out_idx] = total_val;
}

template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_med(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant int* shape [[buffer(4)]],
    const constant size_t* strides [[buffer(5)]],
    const constant int& ndim [[buffer(6)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[dispatch_simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  Op op;
  
  uint out_idx = simd_per_group * tid + simd_group_id;

  if(out_idx >= out_size) {
    return;
  }

  uint in_idx = elem_to_loc(out_idx, shape, strides, ndim);
  in += in_idx;

  U total_val = Op::init;

  for(short i = simd_lane_id; i < short(reduction_size); i+=32) {
    total_val = op(static_cast<U>(in[i]), total_val);
  }

  total_val = op.simd_reduce(total_val);

  if(simd_lane_id == 0) {
    out[out_idx] = total_val;
  }
}

#define instantiate_row_reduce_small(name, itype, otype, op) \
  template[[host_name("row_reduce_general_small_" #name)]] \
  [[kernel]] void row_reduce_general_small<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device otype *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& out_size [[buffer(3)]], \
      const constant int* shape [[buffer(4)]], \
      const constant size_t* strides [[buffer(5)]], \
      const constant int& ndim [[buffer(6)]], \
      uint lid [[thread_position_in_grid]]); \
  template[[host_name("row_reduce_general_med_" #name)]] \
  [[kernel]] void row_reduce_general_med<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device otype *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& out_size [[buffer(3)]], \
      const constant int* shape [[buffer(4)]], \
      const constant size_t* strides [[buffer(5)]], \
      const constant int& ndim [[buffer(6)]], \
      uint tid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_per_group [[dispatch_simdgroups_per_threadgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_row_reduce_general(name, itype, otype, op) \
  instantiate_row_reduce_small(name, itype, otype, op) \
  template [[host_name("row_reduce_general_" #name)]] \
  [[kernel]] void row_reduce_general<itype, otype, op>( \
      const device itype *in [[buffer(0)]],  \
      device mlx_atomic<otype> *out [[buffer(1)]],  \
      const constant size_t& reduction_size [[buffer(2)]],  \
      const constant size_t& out_size [[buffer(3)]],  \
      const constant int* shape [[buffer(4)]],  \
      const constant size_t* strides [[buffer(5)]],  \
      const constant int& ndim [[buffer(6)]],  \
      uint3 lid [[thread_position_in_threadgroup]],  \
      uint3 lsize [[threads_per_threadgroup]],  \
      uint3 tid [[threadgroup_position_in_grid]],  \
      uint simd_lane_id [[thread_index_in_simdgroup]],  \
      uint simd_per_group [[simdgroups_per_threadgroup]],  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

#define instantiate_row_reduce_general_no_atomics(name, itype, otype, op) \
  instantiate_row_reduce_small(name, itype, otype, op) \
  template [[host_name("row_reduce_general_no_atomics_" #name)]] \
  [[kernel]] void row_reduce_general_no_atomics<itype, otype, op>( \
      const device itype *in [[buffer(0)]],  \
      device otype *out [[buffer(1)]],  \
      const constant size_t& reduction_size [[buffer(2)]],  \
      const constant size_t& out_size [[buffer(3)]],  \
      const constant int* shape [[buffer(4)]],  \
      const constant size_t* strides [[buffer(5)]],  \
      const constant int& ndim [[buffer(6)]],  \
      uint3 lid [[thread_position_in_threadgroup]],  \
      uint3 lsize [[threads_per_threadgroup]],  \
      uint3 gsize [[threads_per_grid]], \
      uint3 tid [[threadgroup_position_in_grid]],  \
      uint simd_lane_id [[thread_index_in_simdgroup]],  \
      uint simd_per_group [[simdgroups_per_threadgroup]],  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);


#define instantiate_same_row_reduce_helper(name, tname, type, op) \
  instantiate_row_reduce_general(name ##tname, type, type, op<type>)

#define instantiate_same_row_reduce_na_helper(name, tname, type, op) \
  instantiate_row_reduce_general_no_atomics(name ##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_row_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_row_reduce_na_helper, instantiate_reduce_helper_64b)


instantiate_reduce_from_types(instantiate_row_reduce_general, and, bool, And)
instantiate_reduce_from_types(instantiate_row_reduce_general, or, bool, Or)

instantiate_row_reduce_general(sumbool_, bool, uint32_t, Sum<uint32_t>)