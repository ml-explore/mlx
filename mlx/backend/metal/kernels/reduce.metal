#include <metal_atomic>
#include <metal_simdgroup>

#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/kernels/reduce.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

static constant uint8_t simd_size = 32;

template <typename T, typename Op>
[[kernel]] void init_reduce(
    device T *out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
  out[tid] = Op::init;
}

#define instantiate_init_reduce(name, otype, op) \
  template [[host_name("i" #name)]] \
    [[kernel]] void init_reduce<otype, op>( \
      device otype *out [[buffer(1)]], \
      uint tid [[thread_position_in_grid]]);


///////////////////////////////////////////////////////////////////////////////
// All reduce
///////////////////////////////////////////////////////////////////////////////

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
  // NB: this kernel assumes threads_per_threadgroup is at most
  // 1024. This way with a simd_size of 32, we are guaranteed to
  // complete the reduction in two steps of simd-level reductions.

  Op op;
  threadgroup U local_vals[simd_size];
      
  U total_val = Op::init;

  in += gid * N_READS;
      
  int r = 0;
  for(; r < (int)ceildiv(in_size, grid_size * N_READS) - 1; r++) {
    U vals[N_READS] = {op.init}; 

    for(int i = 0; i < N_READS; i++) {
      vals[i] = static_cast<U>(in[i]);
    }
    for(int i = 0; i < N_READS; i++) {
      total_val = op(vals[i], total_val);
    }

    in += grid_size * N_READS;
  }

  // Sepate case for the last set as we close the reduction size 
  size_t curr_idx = (gid + r * (size_t)grid_size) * N_READS;
  if (curr_idx < in_size) {
    int max_reads = in_size - curr_idx;
    T vals[N_READS];

    for(int i = 0, idx = 0; i < N_READS; i++, idx++) {
      idx = idx < max_reads ? idx : max_reads - 1;
      vals[i] = in[idx];
    }
    for(int i = 0; i < N_READS; i++) {
      U val = i < max_reads ? vals[i] : Op::init; 
      total_val = op(static_cast<U>(val), total_val);
    }
  }

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


///////////////////////////////////////////////////////////////////////////////
// General reduce
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op>
[[kernel]] void general_reduce(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const device int *in_shape [[buffer(2)]],
    const device size_t *in_strides [[buffer(3)]],
    const device size_t *out_strides [[buffer(4)]],
    const device size_t& ndim [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
  Op op;
  auto in_idx = elem_to_loc(gid, in_shape, in_strides, ndim);
  auto out_idx = elem_to_loc(gid, in_shape, out_strides, ndim);
  op.atomic_update(out, static_cast<U>(in[in_idx]), out_idx);
}

template <typename T, typename U, typename Op, int NDIM>
[[kernel]] void general_reduce(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const device int *in_shape [[buffer(2)]],
    const device size_t *in_strides [[buffer(3)]],
    const device size_t *out_strides [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  Op op;
  auto in_idx = elem_to_loc_nd<NDIM>(gid, in_shape, in_strides);
  auto out_idx = elem_to_loc_nd<NDIM>(gid, in_shape, out_strides);
  op.atomic_update(out, static_cast<U>(in[in_idx]), out_idx);
}

#define instantiate_general_reduce_helper(name, itype, otype, op) \
  template [[host_name("general_reduce_" #name)]] \
  [[kernel]] void general_reduce<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device mlx_atomic<otype> *out [[buffer(1)]], \
      const device int *in_shape [[buffer(2)]], \
      const device size_t *in_strides [[buffer(3)]], \
      const device size_t *out_strides [[buffer(4)]], \
      const device size_t& ndim [[buffer(5)]], \
      uint gid [[thread_position_in_grid]]);

#define instantiate_general_reduce_helper_nd(name, itype, otype, op, n) \
  template [[host_name("general_reduce_" #name "_dim_" #n)]] \
  [[kernel]] void general_reduce<itype, otype, op, n>( \
      const device itype *in [[buffer(0)]], \
      device mlx_atomic<otype> *out [[buffer(1)]], \
      const device int *in_shape [[buffer(2)]], \
      const device size_t *in_strides [[buffer(3)]], \
      const device size_t *out_strides [[buffer(4)]], \
      uint gid [[thread_position_in_grid]]);

#define instantiate_general_reduce(name, itype, otype, op) \
  instantiate_general_reduce_helper(name, itype, otype, op) \
  instantiate_general_reduce_helper_nd(name, itype, otype, op, 1) \
  instantiate_general_reduce_helper_nd(name, itype, otype, op, 2) \
  instantiate_general_reduce_helper_nd(name, itype, otype, op, 3) \
  instantiate_general_reduce_helper_nd(name, itype, otype, op, 4)


///////////////////////////////////////////////////////////////////////////////
// Row atomics
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS=REDUCE_N_READS>
[[kernel]] void row_reduce(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
    const device size_t& reduction_size [[buffer(2)]],
    uint lid [[thread_position_in_threadgroup]],
    uint lsize [[threads_per_threadgroup]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_per_group [[simdgroups_per_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {

  Op op;

  // Each threadgroup handles 1 reduction 
  in += tid * reduction_size + lid * N_READS;
  
  // The reduction is accumulated here
  U total_val = Op::init;
  threadgroup U local_vals[simd_size];

  // Loop over the reduction size within thread group
  int r = 0;
  for (; r < (int)ceildiv(reduction_size, N_READS*lsize) - 1; r++) {
    T vals[N_READS]; 
    for(int i = 0; i < N_READS; i++) {
      vals[i] = in[i];
    }
    for(int i = 0; i < N_READS; i++) {
      total_val = op(static_cast<U>(vals[i]), total_val);
    }

    in += lsize * N_READS;
  }

  // Sepate case for the last set as we close the reduction size   
  size_t reduction_index = (lid + (size_t)lsize * r) * N_READS;
  if(reduction_index < reduction_size) {
    int max_reads = reduction_size - reduction_index;

    T vals[N_READS]; 
    for(int i = 0; i < N_READS; i++) {
      int idx = min(i, max_reads - 1);
      vals[i] = static_cast<U>(in[idx]);
    }
    for(int i = 0; i < N_READS; i++) {
      T val = i < max_reads ? vals[i] : Op::init;
      total_val = op(static_cast<U>(val), total_val);
    }
  }

  total_val = op.simd_reduce(total_val);
  
  // Prepare next level
  if (simd_lane_id == 0) {
    local_vals[simd_group_id] = total_val;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
      
  // Reduction within thread group
  //    Only needed if multiple simd groups
  if(reduction_size > simd_size) {
    total_val = lid < simd_per_group ? local_vals[lid] : op.init;
    total_val = op.simd_reduce(total_val);
  }
  // Update output
  if (lid == 0) {
    out[tid] = total_val;
  }
}

#define instantiate_row_reduce(name, itype, otype, op) \
  template [[host_name("row_reduce_" #name)]] \
  [[kernel]] void row_reduce<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device otype *out [[buffer(1)]], \
      const device size_t& reduction_size [[buffer(2)]], \
      uint lid [[thread_position_in_threadgroup]], \
      uint lsize [[threads_per_threadgroup]], \
      uint tid [[threadgroup_position_in_grid]], \
      uint simd_lane_id [[thread_index_in_simdgroup]], \
      uint simd_per_group [[simdgroups_per_threadgroup]], \
      uint simd_group_id [[simdgroup_index_in_threadgroup]]);


///////////////////////////////////////////////////////////////////////////////
// Column reduce
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
inline void _contiguous_strided_reduce(
    const device T *in, 
    device mlx_atomic<U> *out, 
    threadgroup U *local_data, 
    uint in_idx, 
    uint out_idx, 
    uint reduction_size, 
    uint reduction_stride, 
    uint2 tid, 
    uint2 lid, 
    uint2 lsize) {

  Op op;
  T local_vals[N_READS]; 

  uint base_offset = (tid.y * lsize.y + lid.y) * N_READS;

  for(uint r = 0; r < N_READS; r++) {
    uint offset = base_offset + r; 
    offset = offset < reduction_size ? offset : reduction_size - 1; 
    local_vals[r] = in[in_idx + offset * reduction_stride];
  }

  U total_val = Op::init; 
  for(uint r = 0; r < N_READS && (base_offset + r) < reduction_size; r++) {
    total_val = op(static_cast<U>(total_val), local_vals[r]); 
  }
  local_data[lsize.y * lid.x + lid.y] = total_val; 

  threadgroup_barrier(mem_flags::mem_threadgroup);

  if(lid.y == 0) {
    U val = op.init; 

    for(uint i = 0; i < lsize.y; i++) {
      val = op(val, local_data[lsize.y * lid.x + i]); 
    }

    op.atomic_update(out, val, out_idx);
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]], 
    threadgroup U *local_data [[threadgroup(0)]],
    uint2 tid [[threadgroup_position_in_grid]], 
    uint2 lid [[thread_position_in_threadgroup]], 
    uint2 lsize [[threads_per_threadgroup]]) {
  auto out_idx = tid.x * lsize.x + lid.x;  
  
  if(out_idx < out_size) {
    _contiguous_strided_reduce<T, U, Op, N_READS>(
      in, 
      out, 
      local_data, 
      out_idx, 
      out_idx, 
      reduction_size, 
      reduction_stride, 
      tid, 
      lid, 
      lsize); 
  }
}

#define instantiate_col_reduce(name, itype, otype, op) \
  template [[host_name("col_reduce_" #name)]] \
  [[kernel]] void col_reduce<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device mlx_atomic<otype> *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& reduction_stride [[buffer(3)]], \
      const constant size_t& out_size [[buffer(4)]], \
      threadgroup otype *local_data [[threadgroup(0)]], \
      uint2 tid [[threadgroup_position_in_grid]], \
      uint2 lid [[thread_position_in_threadgroup]], \
      uint2 lsize [[threads_per_threadgroup]]);

template <typename T, typename U, typename Op, int NDIM, int N_READS = 16>
[[kernel]] void contiguous_strided_reduce(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]], 
    const device int* in_shape [[buffer(5)]],
    const device size_t* in_strides [[buffer(6)]],
    threadgroup U *local_data [[threadgroup(0)]],
    uint2 tid [[threadgroup_position_in_grid]], 
    uint2 lid [[thread_position_in_threadgroup]], 
    uint2 lsize [[threads_per_threadgroup]]) {
  
  auto out_idx = tid.x * lsize.x + lid.x; 
  auto in_idx = elem_to_loc_nd<NDIM>(out_idx, in_shape, in_strides);
  
  if(out_idx < out_size) {
    _contiguous_strided_reduce<T, U, Op, N_READS>(
      in, 
      out, 
      local_data, 
      in_idx, 
      out_idx, 
      reduction_size, 
      reduction_stride, 
      tid, 
      lid, 
      lsize); 
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void contiguous_strided_reduce(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]], 
    const device int* in_shape [[buffer(5)]],
    const device size_t* in_strides [[buffer(6)]],
    const device size_t& in_dim [[buffer(7)]], 
    threadgroup U *local_data [[threadgroup(0)]],
    uint2 tid [[threadgroup_position_in_grid]], 
    uint2 lid [[thread_position_in_threadgroup]], 
    uint2 lsize [[threads_per_threadgroup]]) {
  
  auto out_idx = tid.x * lsize.x + lid.x; 
  auto in_idx = elem_to_loc(out_idx, in_shape, in_strides, in_dim);
  
  if(out_idx < out_size) {
    _contiguous_strided_reduce<T, U, Op, N_READS>(
      in, 
      out, 
      local_data, 
      in_idx, 
      out_idx, 
      reduction_size, 
      reduction_stride, 
      tid, 
      lid, 
      lsize); 
  }
}

#define instantiate_contiguous_strided_helper(name, itype, otype, op) \
  template [[host_name("contiguous_strided_reduce_" #name)]] \
  [[kernel]] void contiguous_strided_reduce<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device mlx_atomic<otype> *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& reduction_stride [[buffer(3)]], \
      const constant size_t& out_size [[buffer(4)]], \
      const device int* in_shape [[buffer(5)]], \
      const device size_t* in_strides [[buffer(6)]], \
      const device size_t& in_dim [[buffer(7)]], \
      threadgroup otype *local_data [[threadgroup(0)]], \
      uint2 tid [[threadgroup_position_in_grid]], \
      uint2 lid [[thread_position_in_threadgroup]], \
      uint2 lsize [[threads_per_threadgroup]]);

#define instantiate_contiguous_strided_helper_nd(name, itype, otype, op, n) \
  template [[host_name("contiguous_strided_reduce_" #name "_dim_" #n)]] \
  [[kernel]] void contiguous_strided_reduce<itype, otype, op, n>( \
      const device itype *in [[buffer(0)]], \
      device mlx_atomic<otype> *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& reduction_stride [[buffer(3)]], \
      const constant size_t& out_size [[buffer(4)]], \
      const device int* in_shape [[buffer(5)]], \
      const device size_t* in_strides [[buffer(6)]], \
      threadgroup otype *local_data [[threadgroup(0)]], \
      uint2 tid [[threadgroup_position_in_grid]],  \
      uint2 lid [[thread_position_in_threadgroup]],  \
      uint2 lsize [[threads_per_threadgroup]]);

#define instantiate_contiguous_strided(name, itype, otype, op) \
  instantiate_contiguous_strided_helper(name, itype, otype, op) \
  instantiate_contiguous_strided_helper_nd(name, itype, otype, op, 1) \
  instantiate_contiguous_strided_helper_nd(name, itype, otype, op, 2) \
  instantiate_contiguous_strided_helper_nd(name, itype, otype, op, 3) \
  instantiate_contiguous_strided_helper_nd(name, itype, otype, op, 4)


///////////////////////////////////////////////////////////////////////////////
// Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_reduce(name, itype, otype, op) \
  instantiate_all_reduce(name, itype, otype, op) \
  instantiate_row_reduce(name, itype, otype, op) \
  instantiate_col_reduce(name, itype, otype, op) \
  instantiate_contiguous_strided(name, itype, otype, op) \
  instantiate_general_reduce(name, itype, otype, op)

#define instantiate_same_reduce(name, tname, type, op) \
  instantiate_init_reduce(name ##tname, type, op<type>) \
  instantiate_reduce(name ##tname, type, type, op<type>)

#define instantiate_reduce_from_types_helper(name, tname, itype, otype, op) \
  instantiate_reduce(name ##tname, itype, otype, op)

#define instantiate_reduce_from_types(name, otype, op) \
  instantiate_reduce_from_types_helper(name, bool_, bool, otype, op) \
  instantiate_reduce_from_types_helper(name, uint8, uint8_t, otype, op) \
  instantiate_reduce_from_types_helper(name, uint16, uint16_t, otype, op) \
  instantiate_reduce_from_types_helper(name, uint32, uint32_t, otype, op) \
  instantiate_reduce_from_types_helper(name, int8, int8_t, otype, op) \
  instantiate_reduce_from_types_helper(name, int16, int16_t, otype, op) \
  instantiate_reduce_from_types_helper(name, int32, int32_t, otype, op) \
  instantiate_reduce_from_types_helper(name, int64, int64_t, otype, op) \
  instantiate_reduce_from_types_helper(name, float16, half, otype, op) \
  instantiate_reduce_from_types_helper(name, float32, float, otype, op) \
  instantiate_reduce_from_types_helper(name, bfloat16, bfloat16_t, otype, op)

// special case bool with larger output type
instantiate_reduce(sumbool_, bool, uint32_t, Sum<uint32_t>)
instantiate_same_reduce(sum, uint8, uint8_t, Sum)
instantiate_same_reduce(sum, uint16, uint16_t, Sum)
instantiate_same_reduce(sum, uint32, uint32_t, Sum)
instantiate_same_reduce(sum, int8, int8_t, Sum)
instantiate_same_reduce(sum, int16, int16_t, Sum)
instantiate_same_reduce(sum, int32, int32_t, Sum)
instantiate_same_reduce(sum, float16, half, Sum)
instantiate_same_reduce(sum, float32, float, Sum)

instantiate_same_reduce(prod, uint8, uint8_t, Prod)
instantiate_same_reduce(prod, uint16, uint16_t, Prod)
instantiate_same_reduce(prod, uint32, uint32_t, Prod)
instantiate_same_reduce(prod, int8, int8_t, Prod)
instantiate_same_reduce(prod, int16, int16_t, Prod)
instantiate_same_reduce(prod, int32, int32_t, Prod)
instantiate_same_reduce(prod, float16, half, Prod)
instantiate_same_reduce(prod, float32, float, Prod)

instantiate_same_reduce(sum, bfloat16, bfloat16_t, Sum)
instantiate_same_reduce(prod, bfloat16, bfloat16_t, Prod)

instantiate_init_reduce(andbool_, bool, And)
instantiate_reduce_from_types(and, bool, And)

instantiate_init_reduce(orbool_, bool, Or)
instantiate_reduce_from_types(or, bool, Or)

// Compiler segfaulted with the names "min" or "max" ...
instantiate_same_reduce(min_, uint8, uint8_t, Min)
instantiate_same_reduce(min_, uint16, uint16_t, Min)
instantiate_same_reduce(min_, uint32, uint32_t, Min)
instantiate_same_reduce(min_, int8, int8_t, Min)
instantiate_same_reduce(min_, int16, int16_t, Min)
instantiate_same_reduce(min_, int32, int32_t, Min)
instantiate_same_reduce(min_, float16, half, Min)
instantiate_same_reduce(min_, float32, float, Min)

instantiate_same_reduce(max_, uint8, uint8_t, Max)
instantiate_same_reduce(max_, uint16, uint16_t, Max)
instantiate_same_reduce(max_, uint32, uint32_t, Max)
instantiate_same_reduce(max_, int8, int8_t, Max)
instantiate_same_reduce(max_, int16, int16_t, Max)
instantiate_same_reduce(max_, int32, int32_t, Max)
instantiate_same_reduce(max_, float16, half, Max)
instantiate_same_reduce(max_, float32, float, Max)

instantiate_same_reduce(min_, bfloat16, bfloat16_t, Min)
instantiate_same_reduce(max_, bfloat16, bfloat16_t, Max)