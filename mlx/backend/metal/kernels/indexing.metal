// Copyright © 2023 Apple Inc.

#include <metal_atomic>
#include <metal_texture>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/reduce.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

/////////////////////////////////////////////////////////////////////
// Gather kernel
/////////////////////////////////////////////////////////////////////

template <typename IdxT, int NIDX>
struct Indices {
  const array<device IdxT*, NIDX> buffers [[id(0)]];
  device int* shapes [[id(NIDX + 1)]];
  device size_t* strides [[id(NIDX + 2)]];
  const int ndim [[id(NIDX + 3)]];
};

template <typename IdxT>
inline size_t offset_neg_idx(IdxT idx, size_t size) {
  return (idx < 0) ? idx + size : idx;
}

template <>
inline size_t offset_neg_idx(bool idx, size_t) {
  return idx;
}

template <>
inline size_t offset_neg_idx(uint32_t idx, size_t) {
  return idx;
}

template <typename T, typename IdxT, int NIDX>
[[kernel]] void gather(
    const device T *src [[buffer(0)]],
    const device Indices<IdxT, NIDX>& indices [[buffer(1)]],
    device T *out [[buffer(2)]],
    const device int *src_shape [[buffer(3)]],
    const device size_t *src_strides [[buffer(4)]],
    const device size_t& src_ndim [[buffer(5)]],
    const device int *slice_sizes [[buffer(6)]],
    const device size_t& slice_size [[buffer(7)]],
    const device int *axes [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {

  auto ind_idx = gid / slice_size;
  auto ind_offset = gid % slice_size;

  size_t src_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    auto idx_loc = elem_to_loc(
        ind_idx,
        &indices.shapes[indices.ndim * i],
        &indices.strides[indices.ndim * i],
        indices.ndim);
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(
        indices.buffers[i][idx_loc], src_shape[ax]);
    src_idx += idx_val * src_strides[ax];
  }

  auto src_offset = elem_to_loc(
      ind_offset, slice_sizes, src_strides, src_ndim);
  out[gid] = src[src_idx + src_offset];
}

#define instantiate_gather4(name, src_type, ind_type, nindex) \
template [[host_name("gather" name "_" #nindex)]] \
[[kernel]] void gather( \
    const device src_type *src [[buffer(0)]], \
    const device Indices<ind_type, nindex>& indices [[buffer(1)]], \
    device src_type *out [[buffer(2)]], \
    const device int *src_shape [[buffer(3)]], \
    const device size_t *src_strides [[buffer(4)]], \
    const device size_t& src_ndim [[buffer(5)]], \
    const device int *slice_sizes [[buffer(6)]], \
    const device size_t& slice_size [[buffer(7)]], \
    const device int* axes [[buffer(8)]], \
    uint gid [[thread_position_in_grid]]);

// Special for case NIDX=0
instantiate_gather4("bool_", bool, bool, 0)
instantiate_gather4("uint8", uint8_t, bool, 0)
instantiate_gather4("uint16", uint16_t, bool, 0)
instantiate_gather4("uint32", uint32_t, bool, 0)
instantiate_gather4("uint64", uint64_t, bool, 0)
instantiate_gather4("int8", int8_t, bool, 0)
instantiate_gather4("int16", int16_t, bool, 0)
instantiate_gather4("int32", int32_t, bool, 0)
instantiate_gather4("int64", int64_t, bool, 0)
instantiate_gather4("float16", half, bool, 0)
instantiate_gather4("float32", float, bool, 0)
instantiate_gather4("bfloat16", bfloat16_t, bool, 0)

#define instantiate_gather3(name, src_type, ind_type) \
  instantiate_gather4(name, src_type, ind_type, 1) \
  instantiate_gather4(name, src_type, ind_type, 2) \
  instantiate_gather4(name, src_type, ind_type, 3) \
  instantiate_gather4(name, src_type, ind_type, 4) \
  instantiate_gather4(name, src_type, ind_type, 5) \
  instantiate_gather4(name, src_type, ind_type, 6) \
  instantiate_gather4(name, src_type, ind_type, 7) \
  instantiate_gather4(name, src_type, ind_type, 8) \
  instantiate_gather4(name, src_type, ind_type, 9) \
  instantiate_gather4(name, src_type, ind_type, 10)

#define instantiate_gather(name, src_type) \
  instantiate_gather3(#name "bool_", src_type, bool) \
  instantiate_gather3(#name "uint8", src_type, uint8_t) \
  instantiate_gather3(#name "uint16", src_type, uint16_t) \
  instantiate_gather3(#name "uint32", src_type, uint32_t) \
  instantiate_gather3(#name "uint64", src_type, uint64_t) \
  instantiate_gather3(#name "int8", src_type, int8_t) \
  instantiate_gather3(#name "int16", src_type, int16_t) \
  instantiate_gather3(#name "int32", src_type, int32_t) \
  instantiate_gather3(#name "int64", src_type, int64_t)

instantiate_gather(bool_, bool)
instantiate_gather(uint8, uint8_t)
instantiate_gather(uint16, uint16_t)
instantiate_gather(uint32, uint32_t)
instantiate_gather(uint64, uint64_t)
instantiate_gather(int8, int8_t)
instantiate_gather(int16, int16_t)
instantiate_gather(int32, int32_t)
instantiate_gather(int64, int64_t)
instantiate_gather(float16, half)
instantiate_gather(float32, float)
instantiate_gather(bfloat16, bfloat16_t)

/////////////////////////////////////////////////////////////////////
// Scatter kernel
/////////////////////////////////////////////////////////////////////

template <typename T, typename IdxT, typename Op, int NIDX>
[[kernel]] void scatter(
    const device Indices<IdxT, NIDX>& indices [[buffer(0)]],
    const device T *updates [[buffer(1)]],
    device mlx_atomic<T> *out [[buffer(2)]],
    const device int *upd_shape [[buffer(3)]],
    const device size_t *upd_strides [[buffer(4)]],
    const device size_t& upd_ndim [[buffer(5)]],
    const device size_t& upd_size [[buffer(6)]],
    const device int *out_shape [[buffer(7)]],
    const device size_t *out_strides [[buffer(8)]],
    const device size_t& out_ndim [[buffer(9)]],
    const device int* axes [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {

  Op op;
  auto ind_idx = gid / upd_size;
  auto ind_offset = gid % upd_size;

  size_t out_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    auto idx_loc = elem_to_loc(
        ind_idx,
        &indices.shapes[indices.ndim * i],
        &indices.strides[indices.ndim * i],
        indices.ndim);
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(
        indices.buffers[i][idx_loc], out_shape[ax]);
    out_idx += idx_val * out_strides[ax];
  }

  auto out_offset = elem_to_loc(
      ind_offset, upd_shape + indices.ndim, out_strides, out_ndim);
  auto upd_idx = elem_to_loc(gid, upd_shape, upd_strides, upd_ndim);
  op.atomic_update(out, updates[upd_idx], out_idx + out_offset);
}

#define instantiate_scatter4(name, type, ind_type, op_type, nindex) \
template [[host_name("scatter" name "_" #nindex)]] \
[[kernel]] void scatter<type, ind_type, op_type, nindex>( \
    const device Indices<ind_type, nindex>& indices [[buffer(0)]], \
    const device type *updates [[buffer(1)]], \
    device mlx_atomic<type> *out [[buffer(2)]], \
    const device int *upd_shape [[buffer(3)]], \
    const device size_t *upd_strides [[buffer(4)]], \
    const device size_t& upd_ndim [[buffer(5)]], \
    const device size_t& upd_size [[buffer(6)]], \
    const device int *out_shape [[buffer(7)]], \
    const device size_t *out_strides [[buffer(8)]], \
    const device size_t& out_ndim [[buffer(9)]], \
    const device int* axes [[buffer(10)]], \
    uint gid [[thread_position_in_grid]]);

// Special case NINDEX=0
#define instantiate_scatter_nd0(name, type) \
  instantiate_scatter4(#name "none", type, bool, None, 0) \
  instantiate_scatter4(#name "_sum", type, bool, Sum<type>, 0) \
  instantiate_scatter4(#name "_prod", type, bool, Prod<type>, 0) \
  instantiate_scatter4(#name "_max", type, bool, Max<type>, 0) \
  instantiate_scatter4(#name "_min", type, bool, Min<type>, 0)

#define instantiate_scatter3(name, type, ind_type, op_type) \
  instantiate_scatter4(name, type, ind_type, op_type, 1) \
  instantiate_scatter4(name, type, ind_type, op_type, 2) \
  instantiate_scatter4(name, type, ind_type, op_type, 3) \
  instantiate_scatter4(name, type, ind_type, op_type, 4) \
  instantiate_scatter4(name, type, ind_type, op_type, 5) \
  instantiate_scatter4(name, type, ind_type, op_type, 6) \
  instantiate_scatter4(name, type, ind_type, op_type, 7) \
  instantiate_scatter4(name, type, ind_type, op_type, 8) \
  instantiate_scatter4(name, type, ind_type, op_type, 9) \
  instantiate_scatter4(name, type, ind_type, op_type, 10)

#define instantiate_scatter2(name, type, ind_type) \
  instantiate_scatter3(name "_none", type, ind_type, None) \
  instantiate_scatter3(name "_sum", type, ind_type, Sum<type>) \
  instantiate_scatter3(name "_prod", type, ind_type, Prod<type>) \
  instantiate_scatter3(name "_max", type, ind_type, Max<type>) \
  instantiate_scatter3(name "_min", type, ind_type, Min<type>)

#define instantiate_scatter(name, type) \
  instantiate_scatter2(#name "bool_", type, bool) \
  instantiate_scatter2(#name "uint8", type, uint8_t) \
  instantiate_scatter2(#name "uint16", type, uint16_t) \
  instantiate_scatter2(#name "uint32", type, uint32_t) \
  instantiate_scatter2(#name "uint64", type, uint64_t) \
  instantiate_scatter2(#name "int8", type, int8_t) \
  instantiate_scatter2(#name "int16", type, int16_t) \
  instantiate_scatter2(#name "int32", type, int32_t) \
  instantiate_scatter2(#name "int64", type, int64_t)

// TODO uint64 and int64 unsupported
instantiate_scatter_nd0(bool_, bool)
instantiate_scatter_nd0(uint8, uint8_t)
instantiate_scatter_nd0(uint16, uint16_t)
instantiate_scatter_nd0(uint32, uint32_t)
instantiate_scatter_nd0(int8, int8_t)
instantiate_scatter_nd0(int16, int16_t)
instantiate_scatter_nd0(int32, int32_t)
instantiate_scatter_nd0(float16, half)
instantiate_scatter_nd0(float32, float)
instantiate_scatter_nd0(bfloat16, bfloat16_t)

instantiate_scatter(bool_, bool)
instantiate_scatter(uint8, uint8_t)
instantiate_scatter(uint16, uint16_t)
instantiate_scatter(uint32, uint32_t)
instantiate_scatter(int8, int8_t)
instantiate_scatter(int16, int16_t)
instantiate_scatter(int32, int32_t)
instantiate_scatter(float16, half)
instantiate_scatter(float32, float)
instantiate_scatter(bfloat16, bfloat16_t)
