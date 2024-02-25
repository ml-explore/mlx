// Copyright © 2023-2024 Apple Inc.

#include <metal_atomic>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/indexing.h"
#include "mlx/backend/metal/kernels/utils.h"

using namespace metal;

/////////////////////////////////////////////////////////////////////
// Gather kernel
/////////////////////////////////////////////////////////////////////

template <typename T, typename IdxT, int NIDX, int IDX_NDIM>
METAL_FUNC void gather_impl(
    const device T *src [[buffer(0)]],
    device T *out [[buffer(1)]],
    const constant int *src_shape [[buffer(2)]],
    const constant size_t *src_strides [[buffer(3)]],
    const constant size_t& src_ndim [[buffer(4)]],
    const constant int *slice_sizes [[buffer(5)]],
    const constant int *axes [[buffer(6)]],
    const thread Indices<IdxT, NIDX>& indices,
    uint2 index [[thread_position_in_grid]],
    uint2 grid_dim [[threads_per_grid]]) {

  auto ind_idx = index.x;
  auto ind_offset = index.y;

  size_t src_idx = 0;
  for (int i = 0; i < NIDX; ++i) {
    size_t idx_loc;
    if (IDX_NDIM == 0) {
      idx_loc = 0;
    } else if (IDX_NDIM == 1) {
      idx_loc = ind_idx * indices.strides[indices.ndim * i];
    } else {
      idx_loc = elem_to_loc(
          ind_idx,
          &indices.shapes[indices.ndim * i],
          &indices.strides[indices.ndim * i],
          indices.ndim);
    }
    auto ax = axes[i];
    auto idx_val = offset_neg_idx(
        indices.buffers[i][idx_loc], src_shape[ax]);
    src_idx += idx_val * src_strides[ax];
  }

  auto src_offset = elem_to_loc(
      ind_offset, slice_sizes, src_strides, src_ndim);

  size_t out_idx = index.y + static_cast<size_t>(grid_dim.y) * index.x;
  out[out_idx] = src[src_offset + src_idx];

}

#define make_gather_impl(IDX_ARG, IDX_ARR) \
template <typename T, typename IdxT, int NIDX, int IDX_NDIM>  \
[[kernel]] void gather( \
    const device T *src [[buffer(0)]], \
    device T *out [[buffer(1)]], \
    const constant int *src_shape [[buffer(2)]], \
    const constant size_t *src_strides [[buffer(3)]], \
    const constant size_t& src_ndim [[buffer(4)]], \
    const constant int *slice_sizes [[buffer(5)]], \
    const constant int *axes [[buffer(6)]], \
    const constant int *idx_shapes [[buffer(7)]], \
    const constant size_t *idx_strides [[buffer(8)]], \
    const constant int& idx_ndim [[buffer(9)]], \
    IDX_ARG(IdxT) \
    uint2 index [[thread_position_in_grid]], \
    uint2 grid_dim [[threads_per_grid]]) { \
 \
  Indices<IdxT, NIDX> idxs{ \
      {{IDX_ARR()}}, \
      idx_shapes, \
      idx_strides, \
      idx_ndim}; \
 \
  return gather_impl<T, IdxT, NIDX, IDX_NDIM>( \
      src, \
      out, \
      src_shape, \
      src_strides, \
      src_ndim, \
      slice_sizes, \
      axes, \
      idxs, \
      index, \
      grid_dim); \
} 

#define make_gather(n) make_gather_impl(IDX_ARG_ ##n, IDX_ARR_ ##n)

make_gather(0)
make_gather(1)
make_gather(2)
make_gather(3)
make_gather(4)
make_gather(5)
make_gather(6)
make_gather(7)
make_gather(8)
make_gather(9)
make_gather(10)

/////////////////////////////////////////////////////////////////////
// Gather instantiations
/////////////////////////////////////////////////////////////////////

#define instantiate_gather6(name, src_t, idx_t, nidx, IDX_ARG, nd, nd_name) \
template [[host_name("gather" name "_" #nidx "" #nd_name)]] \
[[kernel]] void gather<src_t, idx_t, nidx, nd>( \
    const device src_t *src [[buffer(0)]], \
    device src_t *out [[buffer(1)]], \
    const constant int *src_shape [[buffer(2)]], \
    const constant size_t *src_strides [[buffer(3)]], \
    const constant size_t& src_ndim [[buffer(4)]], \
    const constant int *slice_sizes [[buffer(5)]], \
    const constant int *axes [[buffer(6)]], \
    const constant int *idx_shapes [[buffer(7)]], \
    const constant size_t *idx_strides [[buffer(8)]], \
    const constant int& idx_ndim [[buffer(9)]], \
    IDX_ARG(idx_t) \
    uint2 index [[thread_position_in_grid]], \
    uint2 grid_dim [[threads_per_grid]]);

#define instantiate_gather5(name, src_t, idx_t, nidx, nd, nd_name) \
  instantiate_gather6(name, src_t, idx_t, nidx, IDX_ARG_ ##nidx, nd, nd_name)

#define instantiate_gather4(name, src_t, idx_t, nidx) \
  instantiate_gather5(name, src_t, idx_t, nidx, 0, _0) \
  instantiate_gather5(name, src_t, idx_t, nidx, 1, _1) \
  instantiate_gather5(name, src_t, idx_t, nidx, 2, )


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