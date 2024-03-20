// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/kernels/reduction/utils.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_inst.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Small row reductions
///////////////////////////////////////////////////////////////////////////////

// Each thread reduces for one output
template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_small(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& out_size [[buffer(3)]],
    const constant size_t& non_row_reductions [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    uint lid [[thread_position_in_grid]]) {

  Op op;
  
  uint out_idx = lid;

  if(out_idx >= out_size) {
    return;
  }

  U total_val = Op::init;

  for(short r = 0; r < short(non_row_reductions); r++) {
    uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
    const device T * in_row = in + in_idx;
    
    for(short i = 0; i < short(reduction_size); i++) {
      total_val = op(static_cast<U>(in_row[i]), total_val);
    }
  }

  out[out_idx] = total_val;
}

// Each simdgroup reduces for one output
template <typename T, typename U, typename Op>
[[kernel]] void row_reduce_general_med(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
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

  if(out_idx >= out_size) {
    return;
  }

  U total_val = Op::init;

  if(short(non_row_reductions) == 1) {
    uint in_idx = elem_to_loc(out_idx, shape, strides, ndim);
    const device T * in_row = in + in_idx;

    for(short i = simd_lane_id; i < short(reduction_size); i += 32) {
      total_val = op(static_cast<U>(in_row[i]), total_val);
    }
  }

  else if (short(non_row_reductions) >= 32) {

    for(short r = simd_lane_id; r < short(non_row_reductions); r+=32) {

      uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
      const device T * in_row = in + in_idx;

      for(short i = 0; i < short(reduction_size); i++) {
        total_val = op(static_cast<U>(in_row[i]), total_val);
      }

    }

  }

  else {

    const short n_reductions = short(reduction_size) * short(non_row_reductions);
    const short reductions_per_thread = (n_reductions + simd_size - 1) / simd_size;

    const short r_st = simd_lane_id / reductions_per_thread;
    const short r_ed = short(non_row_reductions);
    const short r_jump = simd_size / reductions_per_thread;

    const short i_st = simd_lane_id % reductions_per_thread;
    const short i_ed = short(reduction_size);
    const short i_jump = reductions_per_thread;

    if(r_st < r_jump) {
      for(short r = r_st; r < r_ed; r += r_jump) {

        uint in_idx = elem_to_loc(out_idx + r * out_size, shape, strides, ndim);
        const device T * in_row = in + in_idx;

        for(short i = i_st; i < i_ed; i += i_jump) {
          total_val = op(static_cast<U>(in_row[i]), total_val);
        }

      }
    }

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
      const constant size_t& non_row_reductions [[buffer(4)]], \
      const constant int* shape [[buffer(5)]], \
      const constant size_t* strides [[buffer(6)]], \
      const constant int& ndim [[buffer(7)]], \
      uint lid [[thread_position_in_grid]]); \
  template[[host_name("row_reduce_general_med_" #name)]] \
  [[kernel]] void row_reduce_general_med<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device otype *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& out_size [[buffer(3)]], \
      const constant size_t& non_row_reductions [[buffer(4)]], \
      const constant int* shape [[buffer(5)]], \
      const constant size_t* strides [[buffer(6)]], \
      const constant int& ndim [[buffer(7)]], \
      uint tid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_per_group [[dispatch_simdgroups_per_threadgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);

///////////////////////////////////////////////////////////////////////////////
// Large row reductions
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U per_thread_row_reduce(
    const device T* in,
    const constant size_t& reduction_size,
    const constant size_t& out_size,
    const constant int* shape,
    const constant size_t* strides,
    const constant int& ndim,
    uint lsize_x,
    uint lid_x,
    uint2 tid) {
  Op op;

  // Each threadgroup handles 1 reduction
  // TODO: Specializing elem_to_loc would be slightly faster
  int idx = tid.y * out_size + tid.x;
  int extra_offset = elem_to_loc(idx, shape, strides, ndim);
  in += extra_offset + lid_x * N_READS;

  // The reduction is accumulated here
  U total_val = Op::init;

  // Loop over the reduction size within thread group
  int r = 0;
  for (; r < (int)ceildiv(reduction_size, N_READS * lsize_x) - 1; r++) {
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
  size_t reduction_index = (lid_x + (size_t)lsize_x * r) * N_READS;
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

template <typename T, typename U, typename Op, int N_READS=REDUCE_N_READS>
[[kernel]] void row_reduce_general(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
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

#define instantiate_row_reduce_general(name, itype, otype, op) \
  instantiate_row_reduce_small(name, itype, otype, op) \
  template [[host_name("row_reduce_general_" #name)]] \
  [[kernel]] void row_reduce_general<itype, otype, op>( \
      const device itype *in [[buffer(0)]],  \
      device mlx_atomic<otype> *out [[buffer(1)]],  \
      const constant size_t& reduction_size [[buffer(2)]],  \
      const constant size_t& out_size [[buffer(3)]],  \
      const constant size_t& non_row_reductions [[buffer(4)]], \
      const constant int* shape [[buffer(5)]], \
      const constant size_t* strides [[buffer(6)]], \
      const constant int& ndim [[buffer(7)]], \
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
      const constant size_t& non_row_reductions [[buffer(4)]], \
      const constant int* shape [[buffer(5)]], \
      const constant size_t* strides [[buffer(6)]], \
      const constant int& ndim [[buffer(7)]], \
      uint3 lid [[thread_position_in_threadgroup]],  \
      uint3 lsize [[threads_per_threadgroup]],  \
      uint3 gsize [[threads_per_grid]], \
      uint3 tid [[threadgroup_position_in_grid]],  \
      uint simd_lane_id [[thread_index_in_simdgroup]],  \
      uint simd_per_group [[simdgroups_per_threadgroup]],  \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);


///////////////////////////////////////////////////////////////////////////////
// Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_same_row_reduce_helper(name, tname, type, op) \
  instantiate_row_reduce_general(name ##tname, type, type, op<type>)

#define instantiate_same_row_reduce_na_helper(name, tname, type, op) \
  instantiate_row_reduce_general_no_atomics(name ##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_row_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_row_reduce_na_helper, instantiate_reduce_helper_64b)


instantiate_reduce_from_types(instantiate_row_reduce_general, and, bool, And)
instantiate_reduce_from_types(instantiate_row_reduce_general, or, bool, Or)

instantiate_row_reduce_general(sumbool_, bool, uint32_t, Sum<uint32_t>)