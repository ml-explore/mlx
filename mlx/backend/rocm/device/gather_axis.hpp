// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/rocm/device/indexing.hpp"
#include "mlx/backend/rocm/device/utils.hpp"

#include <hip/hip_runtime.h>

namespace mlx::core::rocm {

template <
    typename T,
    typename IdxT,
    int NDIM,
    bool SrcC,
    bool IdxC,
    typename LocT>
__global__ void gather_axis(
    const T* src,
    const IdxT* indices,
    T* out,
    LocT idx_size_pre,
    LocT idx_size_axis,
    LocT idx_size_post,
    const int32_t* shape,
    const int64_t* src_strides,
    const int64_t* idx_strides,
    int32_t axis,
    int32_t axis_size,
    int64_t src_stride_axis,
    int64_t idx_stride_axis) {
  LocT index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= idx_size_pre * idx_size_axis * idx_size_post) {
    return;
  }

  LocT x, y, z;
  index_to_dims(index, idx_size_axis, idx_size_pre, x, y, z);

  LocT elem_idx = z * idx_size_post;

  LocT idx_loc = y * idx_stride_axis;
  if constexpr (IdxC) {
    idx_loc += elem_idx * idx_size_axis + x;
  } else {
    idx_loc += elem_to_loc_nd<NDIM>(elem_idx + x, shape, idx_strides);
  }

  auto idx_val = absolute_index(indices[idx_loc], axis_size);

  LocT src_loc = idx_val * src_stride_axis;
  if constexpr (SrcC) {
    src_loc += elem_idx * axis_size + x;
  } else {
    src_loc += elem_to_loc_nd<NDIM>(elem_idx + x, shape, src_strides);
  }

  LocT out_idx = y * idx_size_post + elem_idx * idx_size_axis + x;

  out[out_idx] = src[src_loc];
}

} // namespace mlx::core::rocm
