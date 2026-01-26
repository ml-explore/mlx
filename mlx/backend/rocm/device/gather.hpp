// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/indexing.hpp"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <typename T, typename IdxT, int NIDX, int IDX_NDIM, typename LocT>
__global__ void gather(
    const T* src,
    T* out,
    LocT size,
    const int32_t* src_shape,
    const int64_t* src_strides,
    int32_t src_ndim,
    const int32_t* slice_sizes,
    uint32_t slice_size,
    const int32_t* axes,
    const IdxT* const* indices,
    const int32_t* indices_shape,
    const int64_t* indices_strides) {
  LocT out_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (out_idx >= size) {
    return;
  }

  LocT src_elem = out_idx % slice_size;
  LocT idx_elem = out_idx / slice_size;

  LocT src_loc = elem_to_loc(src_elem, slice_sizes, src_strides, src_ndim);

#pragma unroll
  for (int i = 0; i < NIDX; ++i) {
    LocT idx_loc = elem_to_loc_nd<IDX_NDIM>(
        idx_elem,
        indices_shape + i * IDX_NDIM,
        indices_strides + i * IDX_NDIM);
    int32_t axis = axes[i];
    LocT idx_val = absolute_index(indices[i][idx_loc], src_shape[axis]);
    src_loc += idx_val * src_strides[axis];
  }

  out[out_idx] = src[src_loc];
}

} // namespace mlx::core::rocm
