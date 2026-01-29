// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/indexing.hpp"
#include "mlx/backend/rocm/device/scatter_ops.hpp"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <
    typename T,
    typename IdxT,
    typename Op,
    int NIDX,
    int IDX_NDIM,
    typename LocT>
__global__ void scatter(
    const T* upd,
    T* out,
    LocT size,
    const int32_t* upd_shape,
    const int64_t* upd_strides,
    int32_t upd_ndim,
    LocT upd_post_idx_size,
    const int32_t* out_shape,
    const int64_t* out_strides,
    int32_t out_ndim,
    const int32_t* axes,
    const IdxT* const* indices,
    const int32_t* indices_shape,
    const int64_t* indices_strides) {
  LocT upd_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (upd_idx >= size) {
    return;
  }

  LocT out_elem = upd_idx % upd_post_idx_size;
  LocT idx_elem = upd_idx / upd_post_idx_size;

  LocT out_idx = elem_to_loc(
      out_elem, upd_shape + IDX_NDIM, out_strides, out_ndim);

#pragma unroll
  for (int i = 0; i < NIDX; ++i) {
    LocT idx_loc = elem_to_loc_nd<IDX_NDIM>(
        idx_elem,
        indices_shape + i * IDX_NDIM,
        indices_strides + i * IDX_NDIM);
    int32_t axis = axes[i];
    LocT idx_val = absolute_index(indices[i][idx_loc], out_shape[axis]);
    out_idx += idx_val * out_strides[axis];
  }

  LocT upd_loc = elem_to_loc(
      out_elem + idx_elem * upd_post_idx_size,
      upd_shape,
      upd_strides,
      upd_ndim);

  Op{}(out + out_idx, upd[upd_loc]);
}

} // namespace mlx::core::rocm
