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
    int NDIM,
    bool UpdC,
    bool IdxC,
    typename LocT = int64_t>
__global__ void scatter_axis_kernel(
    const T* upd,
    const IdxT* indices,
    T* out,
    LocT idx_size_pre,
    LocT idx_size_axis,
    LocT idx_size_post,
    const hip_array<int32_t, MAX_NDIM> shape,
    const hip_array<int64_t, MAX_NDIM> upd_strides,
    const hip_array<int64_t, MAX_NDIM> idx_strides,
    int32_t axis,
    int32_t axis_size,
    int64_t upd_stride_axis,
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
    idx_loc +=
        elem_to_loc_nd<NDIM>(elem_idx + x, shape.data_, idx_strides.data_);
  }

  auto idx_val = absolute_index(indices[idx_loc], axis_size);

  LocT upd_loc = y * upd_stride_axis;
  if constexpr (UpdC) {
    upd_loc += elem_idx * idx_size_axis + x;
  } else {
    upd_loc +=
        elem_to_loc_nd<NDIM>(elem_idx + x, shape.data_, upd_strides.data_);
  }

  LocT out_idx = idx_val * idx_size_post + elem_idx * axis_size + x;

  Op{}(out + out_idx, upd[upd_loc]);
}

} // namespace mlx::core::rocm
