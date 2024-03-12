// Copyright © 2023-2024 Apple Inc.

#include <metal_atomic>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/indexing.h"
#include "mlx/backend/metal/kernels/reduction/ops.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

/////////////////////////////////////////////////////////////////////
// Scatter kernel
/////////////////////////////////////////////////////////////////////

template <typename T, typename IdxT, typename Op, int NIDX> \
METAL_FUNC void scatter_1d_index_impl(
  const device T *updates [[buffer(1)]],
  device mlx_atomic<T> *out [[buffer(2)]],
  const constant int* out_shape [[buffer(3)]],
  const constant size_t* out_strides [[buffer(4)]],
  const constant size_t& upd_size [[buffer(5)]],
  const thread array<const device IdxT*, NIDX>& idx_buffers,
  uint2 gid [[thread_position_in_grid]]) {

  Op op;

  uint out_idx = 0;
  for (int i = 0; i < NIDX; i++) {
    auto idx_val = offset_neg_idx(
        idx_buffers[i][gid.y], out_shape[i]);
    out_idx += idx_val * out_strides[i];
  }

  op.atomic_update(out, updates[gid.y * upd_size + gid.x], out_idx + gid.x);
}

#define make_scatter_1d_index(IDX_ARG, IDX_ARR) \
template <typename T, typename IdxT, typename Op, int NIDX> \
[[kernel]] void scatter_1d_index( \
  const device T *updates [[buffer(1)]], \
  device mlx_atomic<T> *out [[buffer(2)]], \
  const constant int* out_shape [[buffer(3)]], \
  const constant size_t* out_strides [[buffer(4)]], \
  const constant size_t& upd_size [[buffer(5)]], \
  IDX_ARG(IdxT) \
  uint2 gid [[thread_position_in_grid]]) { \
  \
  const array<const device IdxT*, NIDX> idx_buffers = {IDX_ARR()}; \
  \
  return scatter_1d_index_impl<T, IdxT, Op, NIDX>( \
    updates, \
    out, \
    out_shape, \
    out_strides, \
    upd_size, \
    idx_buffers, \
    gid); \
  \
}

template <typename T, typename IdxT, typename Op, int NIDX>
METAL_FUNC void scatter_impl(
    const device T *updates [[buffer(1)]],
    device mlx_atomic<T> *out [[buffer(2)]],
    const constant int *upd_shape [[buffer(3)]],
    const constant size_t *upd_strides [[buffer(4)]],
    const constant size_t& upd_ndim [[buffer(5)]],
    const constant size_t& upd_size [[buffer(6)]],
    const constant int *out_shape [[buffer(7)]],
    const constant size_t *out_strides [[buffer(8)]],
    const constant size_t& out_ndim [[buffer(9)]],
    const constant int* axes [[buffer(10)]],
    const thread Indices<IdxT, NIDX>& indices,
    uint2 gid [[thread_position_in_grid]]) {

  Op op;
  auto ind_idx = gid.y;
  auto ind_offset = gid.x;

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

  if (upd_size > 1) {
    auto out_offset = elem_to_loc(
        ind_offset, upd_shape + indices.ndim, out_strides, out_ndim);
    out_idx += out_offset;
  }

  auto upd_idx = elem_to_loc(gid.y * upd_size + gid.x, upd_shape, upd_strides, upd_ndim);
  op.atomic_update(out, updates[upd_idx], out_idx);
}

#define make_scatter_impl(IDX_ARG, IDX_ARR) \
template <typename T, typename IdxT, typename Op, int NIDX>  \
[[kernel]] void scatter( \
    const device T *updates [[buffer(1)]], \
    device mlx_atomic<T> *out [[buffer(2)]], \
    const constant int *upd_shape [[buffer(3)]], \
    const constant size_t *upd_strides [[buffer(4)]], \
    const constant size_t& upd_ndim [[buffer(5)]], \
    const constant size_t& upd_size [[buffer(6)]], \
    const constant int *out_shape [[buffer(7)]], \
    const constant size_t *out_strides [[buffer(8)]], \
    const constant size_t& out_ndim [[buffer(9)]], \
    const constant int* axes [[buffer(10)]], \
    const constant int *idx_shapes [[buffer(11)]], \
    const constant size_t *idx_strides [[buffer(12)]], \
    const constant int& idx_ndim [[buffer(13)]], \
    IDX_ARG(IdxT) \
    uint2 gid [[thread_position_in_grid]]) { \
 \
  Indices<IdxT, NIDX> idxs{ \
      {{IDX_ARR()}}, \
      idx_shapes, \
      idx_strides, \
      idx_ndim}; \
 \
  return scatter_impl<T, IdxT, Op, NIDX>( \
      updates, \
      out, \
      upd_shape, \
      upd_strides, \
      upd_ndim, \
      upd_size, \
      out_shape, \
      out_strides, \
      out_ndim, \
      axes, \
      idxs, \
      gid); \
}

#define make_scatter(n) \
make_scatter_impl(IDX_ARG_ ##n, IDX_ARR_ ##n) \
make_scatter_1d_index(IDX_ARG_ ##n, IDX_ARR_ ##n)

make_scatter(0)
make_scatter(1)
make_scatter(2)
make_scatter(3)
make_scatter(4)
make_scatter(5)
make_scatter(6)
make_scatter(7)
make_scatter(8)
make_scatter(9)
make_scatter(10)

/////////////////////////////////////////////////////////////////////
// Scatter instantiations
/////////////////////////////////////////////////////////////////////

#define instantiate_scatter5(name, src_t, idx_t, op_t, nidx, IDX_ARG) \
template [[host_name("scatter" name "_" #nidx)]] \
[[kernel]] void scatter<src_t, idx_t, op_t, nidx>( \
    const device src_t *updates [[buffer(1)]], \
    device mlx_atomic<src_t> *out [[buffer(2)]], \
    const constant int *upd_shape [[buffer(3)]], \
    const constant size_t *upd_strides [[buffer(4)]], \
    const constant size_t& upd_ndim [[buffer(5)]], \
    const constant size_t& upd_size [[buffer(6)]], \
    const constant int *out_shape [[buffer(7)]], \
    const constant size_t *out_strides [[buffer(8)]], \
    const constant size_t& out_ndim [[buffer(9)]], \
    const constant int* axes [[buffer(10)]], \
    const constant int *idx_shapes [[buffer(11)]], \
    const constant size_t *idx_strides [[buffer(12)]], \
    const constant int& idx_ndim [[buffer(13)]], \
    IDX_ARG(idx_t) \
    uint2 gid [[thread_position_in_grid]]);

#define instantiate_scatter6(name, src_t, idx_t, op_t, nidx, IDX_ARG) \
template [[host_name("scatter_1d_index" name "_" #nidx)]] \
[[kernel]] void scatter_1d_index<src_t, idx_t, op_t, nidx>( \
  const device src_t *updates [[buffer(1)]], \
  device mlx_atomic<src_t> *out [[buffer(2)]], \
  const constant int* out_shape [[buffer(3)]], \
  const constant size_t* out_strides [[buffer(4)]], \
  const constant size_t& upd_size [[buffer(5)]], \
  IDX_ARG(idx_t) \
  uint2 gid [[thread_position_in_grid]]);

#define instantiate_scatter4(name, src_t, idx_t, op_t, nidx) \
  instantiate_scatter5(name, src_t, idx_t, op_t, nidx, IDX_ARG_ ##nidx) \
  instantiate_scatter6(name, src_t, idx_t, op_t, nidx, IDX_ARG_ ##nidx)

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
