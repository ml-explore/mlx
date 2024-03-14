// Copyright © 2023-2024 Apple Inc.

#include "mlx/backend/metal/kernels/reduction/utils.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/reduction/reduce_inst.h"

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
// Small column reduce kernel
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op>
[[kernel]] void col_reduce_small(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
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

  for(uint i = 0; i < non_col_reductions; i++) {
    size_t in_idx = elem_to_loc(i, non_col_shapes, non_col_strides, non_col_ndim);

    for(uint j = 0; j < reduction_size; j++, in_idx += reduction_stride) {
      U val = static_cast<U>(in[in_idx]);
      total_val = op(total_val, val);
    }
  }

  out[out_idx] = total_val;
}

#define instantiate_col_reduce_small(name, itype, otype, op) \
  template [[host_name("col_reduce_small_" #name)]] \
  [[kernel]] void col_reduce_small<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device otype *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& reduction_stride [[buffer(3)]], \
      const constant size_t& out_size [[buffer(4)]], \
      const constant int* shape [[buffer(5)]],  \
      const constant size_t* strides [[buffer(6)]],  \
      const constant int& ndim [[buffer(7)]],  \
      const constant size_t& non_col_reductions [[buffer(8)]], \
      const constant int* non_col_shapes [[buffer(9)]], \
      const constant size_t* non_col_strides [[buffer(10)]], \
      const constant int& non_col_ndim [[buffer(11)]], \
      uint tid [[thread_position_in_grid]]);

///////////////////////////////////////////////////////////////////////////////
// Column reduce helper
///////////////////////////////////////////////////////////////////////////////

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
METAL_FUNC U _contiguous_strided_reduce(
    const device T* in,
    threadgroup U* local_data,
    uint in_idx,
    uint reduction_size,
    uint reduction_stride,
    uint2 tid,
    uint2 lid,
    uint2 lsize) {
  Op op;
  U total_val = Op::init;

  uint base_offset = (tid.y * lsize.y + lid.y) * N_READS;
  for (uint r = 0; r < N_READS && (base_offset + r) < reduction_size; r++) {
    uint offset = base_offset + r;
    total_val =
        op(static_cast<U>(total_val), in[in_idx + offset * reduction_stride]);
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

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_general(
    const device T *in [[buffer(0)]],
    device mlx_atomic<U> *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup U *local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 lsize [[threads_per_threadgroup]]) {
  auto out_idx = tid.x * lsize.x + lid.x;
  auto in_idx = elem_to_loc(
    out_idx + tid.z * out_size,
    shape,
    strides,
    ndim
  );

  Op op;
  if(out_idx < out_size) {
    U val = _contiguous_strided_reduce<T, U, Op, N_READS>(
              in,
              local_data,
              in_idx,
              reduction_size,
              reduction_stride,
              tid.xy,
              lid.xy,
              lsize.xy);

    // Write out reduction results generated by threadgroups working on specific output element, contiguously.
    if (lid.y == 0) {
      op.atomic_update(out, val, out_idx);
    }
  }
}

template <typename T, typename U, typename Op, int N_READS = REDUCE_N_READS>
[[kernel]] void col_reduce_general_no_atomics(
    const device T *in [[buffer(0)]],
    device U *out [[buffer(1)]],
    const constant size_t& reduction_size [[buffer(2)]],
    const constant size_t& reduction_stride [[buffer(3)]],
    const constant size_t& out_size [[buffer(4)]],
    const constant int* shape [[buffer(5)]],
    const constant size_t* strides [[buffer(6)]],
    const constant int& ndim [[buffer(7)]],
    threadgroup U *local_data [[threadgroup(0)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint3 lsize [[threads_per_threadgroup]],
    uint3 gsize [[threads_per_grid]]) {
  auto out_idx = tid.x * lsize.x + lid.x;
  auto in_idx = elem_to_loc(
    out_idx + tid.z * out_size,
    shape,
    strides,
    ndim
  );

  if(out_idx < out_size) {
    U val = _contiguous_strided_reduce<T, U, Op, N_READS>(
              in,
              local_data,
              in_idx,
              reduction_size,
              reduction_stride,
              tid.xy,
              lid.xy,
              lsize.xy);

    // Write out reduction results generated by threadgroups working on specific output element, contiguously.
    if (lid.y == 0) {
      uint tgsize_y = ceildiv(gsize.y, lsize.y);
      uint tgsize_z = ceildiv(gsize.z, lsize.z);
      out[tgsize_y * tgsize_z * gid.x + tgsize_y * tid.z + tid.y] = val;
    }
  }
}

#define instantiate_col_reduce_general(name, itype, otype, op) \
  template [[host_name("col_reduce_general_" #name)]] \
  [[kernel]] void col_reduce_general<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device mlx_atomic<otype> *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& reduction_stride [[buffer(3)]], \
      const constant size_t& out_size [[buffer(4)]], \
      const constant int* shape [[buffer(5)]],  \
      const constant size_t* strides [[buffer(6)]],  \
      const constant int& ndim [[buffer(7)]],  \
      threadgroup otype *local_data [[threadgroup(0)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]], \
      uint3 lsize [[threads_per_threadgroup]]);

#define instantiate_col_reduce_general_no_atomics(name, itype, otype, op) \
  template [[host_name("col_reduce_general_no_atomics_" #name)]] \
  [[kernel]] void col_reduce_general_no_atomics<itype, otype, op>( \
      const device itype *in [[buffer(0)]], \
      device otype *out [[buffer(1)]], \
      const constant size_t& reduction_size [[buffer(2)]], \
      const constant size_t& reduction_stride [[buffer(3)]], \
      const constant size_t& out_size [[buffer(4)]], \
      const constant int* shape [[buffer(5)]],  \
      const constant size_t* strides [[buffer(6)]],  \
      const constant int& ndim [[buffer(7)]],  \
      threadgroup otype *local_data [[threadgroup(0)]], \
      uint3 tid [[threadgroup_position_in_grid]], \
      uint3 lid [[thread_position_in_threadgroup]], \
      uint3 gid [[thread_position_in_grid]], \
      uint3 lsize [[threads_per_threadgroup]], \
      uint3 gsize [[threads_per_grid]]);

///////////////////////////////////////////////////////////////////////////////
// Instantiations
///////////////////////////////////////////////////////////////////////////////

#define instantiate_same_col_reduce_helper(name, tname, type, op) \
  instantiate_col_reduce_small(name ##tname, type, type, op<type>) \
  instantiate_col_reduce_general(name ##tname, type, type, op<type>)

#define instantiate_same_col_reduce_na_helper(name, tname, type, op) \
  instantiate_col_reduce_small(name ##tname, type, type, op<type>) \
  instantiate_col_reduce_general_no_atomics(name ##tname, type, type, op<type>)

instantiate_reduce_ops(instantiate_same_col_reduce_helper, instantiate_reduce_helper_types)
instantiate_reduce_ops(instantiate_same_col_reduce_na_helper, instantiate_reduce_helper_64b)

instantiate_col_reduce_general(sumbool_, bool, uint32_t, Sum<uint32_t>)
instantiate_reduce_from_types(instantiate_col_reduce_general, and, bool, And)
instantiate_reduce_from_types(instantiate_col_reduce_general, or, bool, Or)

instantiate_col_reduce_small(sumbool_, bool, uint32_t, Sum<uint32_t>)
instantiate_reduce_from_types(instantiate_col_reduce_small, and, bool, And)
instantiate_reduce_from_types(instantiate_col_reduce_small, or, bool, Or)